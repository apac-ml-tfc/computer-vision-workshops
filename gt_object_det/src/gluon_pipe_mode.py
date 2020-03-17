"""Alternative, streaming-ready dataset classes for Gluon

Gluon's core `gluon.data.Dataset` class is wholly predicated on random access: Datasets must
implement the `len()` operator via `__len__()`, and arbitrary indexing via `__getitem__()`.

These classes provide semi-compatible implementations for streaming data with SageMaker Pipe Mode,
under the following assumptions/constraints:

- Streams will be MXNet-native RecordIO format (not TFRecord, or another serialization)
- Streaming datasets are only aware of their length if provided to the constructor, and will raise
  a ValueError if len() is requested when unknown.
- Streaming datasets are designed for sequential access, but may optionally implement memory. They
  may read forward, but not back, and will raise a PipeModeDatumForgotten error if a requested
  datum has already been "forgotten".
- Pipe Mode datasets repeat data through multiple "epochs" (which may be differently ordered), and
  a request for index 0 after it's been forgotten can be assumed as the start of the next epoch.

Loosely, we provide replacement classes for the inheritance chain of GluonCV RecordFileDetection:

- gluoncv.data.RecordFileDetection (see AugmentedManifestDetection)
- mxnet.gluon.data.vision.ImageRecordDataset (see AugmentedManifestImageRecordDataset)
- mxnet.gluon.data.RecordFileDataset (see AugmentedManifestDataset and PipeModeDataset)

"""
# Python Built-Ins:
from collections import deque
import json
import logging
import os
from tempfile import TemporaryDirectory

# External Dependencies:
from gluoncv.data import RecordFileDetection
from mxnet.gluon.data import Dataset
import mxnet.image as image
import mxnet.recordio as recordio
import numpy as np


logger = logging.getLogger("gluon_pipe_mode")

class PipeModeEpochExhausted(ValueError):
    """Error raised when no more data is available in a Pipe Mode Epoch"""
    pass

class PipeModeDatumForgotten(ValueError):
    """Error raised when a requested datum is older than buffered range"""
    pass

class PipeModeEpochReader(object):
    """Gluon Dataset-like reader for a single RecordIO stream file

    Wrapper for a sequential-only-access, non-repeating RecordIO file - which in SageMaker Pipe
    Mode will represent one epoch pass of the channel.
    """
    def __init__(self, filename, memory=1):
        logger.debug(f"New PipeModeEpochReader({filename}, {memory})")
        self.filename = filename
        self._cursor = 0
        self._filebuf = deque(maxlen=memory)
        self.finished = False
        self._reader = recordio.MXRecordIO(filename, "r")

    def read(self):
        logger.debug(f"PipeModeEpochReader.read()")
        newval = self._reader.read()
        if newval is None:
            self.finished = True
            return
        else:
            self._filebuf.append(newval)
            self._cursor += 1
            return newval

    def close(self):
        logger.debug(f"PipeModeEpochReader.close()")
        self._reader.close()

    def __getitem__(self, idx):
        logger.debug(f"PipeModeEpochReader.__getitem__({idx})")
        while idx >= self._cursor and not self.finished:
            self.read()
        if idx >= self._cursor:
            raise PipeModeEpochExhausted(
                f"Index {idx} not in dataset: Exhausted after {self._cursor} items"
            )
        else:
            bufferidx = idx - (self._cursor - len(self._filebuf))
            if bufferidx >= 0:
                return self._filebuf[bufferidx]
            else:
                raise PipeModeDatumForgotten(
                    f"Requested index {idx} is already read and forgotten"
                    f"(Cursor at {self._cursor}, buffer len {len(self._filebuf)})"
                )

    def tell(self):
        logger.debug(f"PipeModeEpochReader.tell()")
        return self._cursor

class PipeModeDataset(Dataset):
    """A dataset wrapping over a SageMaker Pipe Mode RecordIO Stream."""
    def __init__(self, channel_path:str="/opt/ml/input/train", length:int=None):
        logger.debug(f"New PipeModeDataset({channel_path}, {length})")
        self.channel_path = channel_path
        if length is not None:
            assert length >= 0, "length must be a non-negative integer (number of samples in data)"
        self._length = length
        self.epoch = 0
        self.reader = PipeModeEpochReader(f"{channel_path}_{self.epoch}")

    def __getitem__(self, idx):
        logger.debug(f"PipeModeDataset.__getitem__({idx})")
        try:
            return self.reader[idx]
        except PipeModeDatumForgotten as e:
            if (idx == 0):
                self.reader.close()
                self.epoch += 1
                self.reader = PipeModeEpochReader(f"{self.channel_path}_{self.epoch}")
                return self.reader[idx]#self[idx]
            else:
                raise e

    def __len__(self):
        logger.debug(f"PipeModeDataset.__len__()")
        if self._length is None:
            raise ValueError(
                "To query len() of PipeModeDataset, you must supply length argument in constructor"
            )
        else:
            return self._length

class AugmentedManifestDataset(PipeModeDataset):
    """A dataset wrapping for a SageMaker Augmented Manifest Pipe Mode Dataset

    An AugmentedManifest mode input maps a JSON-Lines file to a stream of (RecordIO) records, with
    each datum in the manifest file generating one or more Records: One per attribute.

    For more information, see the SageMaker documentation:

    https://docs.aws.amazon.com/sagemaker/latest/dg/augmented-manifest.html
    """
    def __init__(self, channel_path, n_attributes=2, length=None):
        logger.debug(f"New AugmentedManifestDataset({channel_path}, {n_attributes}, {length})")
        if length is not None:
            super(AugmentedManifestDataset, self).__init__(
                channel_path,
                length=length * n_attributes
            )
        else:
            super(AugmentedManifestDataset, self).__init__(channel_path, length=None)
        self.n_attributes = n_attributes

    def __getitem__(self, idx):
        rawidx = idx * self.n_attributes
        return tuple(
            super(AugmentedManifestDataset, self).__getitem__(i + rawidx)
            for i in range(self.n_attributes)
        )

    def __len__(self):
        return super(AugmentedManifestDataset, self).__len__() // self.n_attributes

class AugmentedManifestImageRecordDataset(AugmentedManifestDataset):
    """A dataset wrapping over an AugmentedManifest channel of images + labels

    Each datum has 2 attributes: The image file contents and the label JSON (binary) string.

    - Image data may be any format supported by mxnet.image.imdecode() - e.g. JPG, etc.
    - Label data is a JSON stringified object (e.g. starting with '{')
    """
    def __init__(self, channel_path, n_attributes=2, length=None, flag=1, transform=None):
        logger.debug(
            f"New AugmentedManifestImageRecordDataset({channel_path}, {n_attributes}, {length}, "
            f"{flag}, {transform})"
        )
        if n_attributes != 2:
            raise NotImplementedError(
                "Currently only 2-attribute datasets (image + annotation) are supported"
            )
        super(AugmentedManifestImageRecordDataset, self).__init__(
            channel_path,
            n_attributes=n_attributes,
            length=length
        )
        self._flag = flag
        self._transform = transform

    def __getitem__(self, idx):
        # We don't assume up-front that annotation and image fields are in a particular (or 
        # guaranteed) order:
        record = super(AugmentedManifestImageRecordDataset, self).__getitem__(idx)
        # TODO: Support non-object annotation fields (e.g. for classification use cases)
        # Boolean list of whether each field is JSON:
        fieldsjson = [field[0] == b"{"[0] for field in record]  # Binary in Python is weird...
        njsonfields = sum(fieldsjson)
        if njsonfields != 1:
            raise ValueError(
                f"Record had {njsonfields} JSON annotation fields out of {len(record)} total: "
                "Expected exactly one"
            )
        # Take first JSON and first non-JSON field to be the header and the image, respectively:
        label = json.loads(record[fieldsjson.index(True)])
        img = record[fieldsjson.index(False)]

        if self._transform is not None:
            return self._transform(image.imdecode(img, self._flag), label)
        return image.imdecode(img, self._flag), label

class AugmentedManifestDetection(AugmentedManifestImageRecordDataset):
    """Object detection dataset for SageMaker AugmentedManifest Pipe Mode channel

    Each datum has 2 attributes: The image file contents and the label JSON (binary) string.

    - Image data may be any format supported by mxnet.image.imdecode() - e.g. JPG, etc.
    - Label data is a JSON stringified object (e.g. starting with '{')
    - Label format is consistent with SageMaker Ground Truth output for a bounding box annotation.

    Note that in SMGT bounding box left/top/width/height props are *raw pixels*, not normalized.

    """
    def __init__(self, channel_path, length=None, flag=1, transform=None):
        logger.debug(
            f"New AugmentedManifestDetection({channel_path}, {length}, {flag}, {transform})"
        )
        super(AugmentedManifestDetection, self).__init__(
            channel_path,
            length=length,
            flag=flag,
            transform=transform
        )

    def __getitem__(self, idx):
        img, label = super(AugmentedManifestDetection, self).__getitem__(idx)
        h, w, _ = img.shape

        # GluonCV internally expects [xmin, ymin, xmax, ymax, id, extra fields...]
        boxes = [[
            ann["left"],
            ann["top"],
            (ann["left"] + ann["width"]),
            (ann["top"] + ann["height"]),
            ann["class_id"],
        ] for ann in label["annotations"]]

        # If there are zero boxes, we need to force a (0, 5) shape:
        boxes = np.array(boxes, ndmin=2)
        if not len(boxes):
            boxes = boxes.reshape((0, 5))
        return img, boxes


def pipe_detection_minibatch(
    epoch:int,
    batch_size:int=50,
    channel:str="/opt/ml/input/data/train",
    discard_partial_final:bool=False
):
    """Legacy generator method for batching RecordFileDetectors from SageMaker Pipe Mode stream

    This procedural method was explored before the cleaner approach of overriding dataset classes.
    The generator reads batches of records from a RecordIO stream, and converts each "stream-batch"
    into a GluonCV RecordFileDetection by buffering the records to an indexed local RecordIO file.

    Pros:

    - Doesn't require a length parameter up-front, because it really iterates through the stream
    - Can be used with shuffling transforms (which isn't necessary as SM can shuffle for you)
    - Total GluonCV compatibility, as it instantiates genuine GluonCV RecordFileDetection class

    Cons:

    - Need to set the stream batch size as a multiple of the minibatch size, which is fiddly
    - Stream is read in batches, which can still block processing on I/O
    - Introduces an outer loop in the training script - not consistent with standard patterns

    Example SageMaker input channel configuration:

    ```
    train_channel = sagemaker.session.s3_input(
        f"s3://{BUCKET_NAME}/{DATA_PREFIX}/train.manifest", # SM Ground Truth output manifest
        content_type="application/x-recordio",
        s3_data_type="AugmentedManifestFile",
        record_wrapping="RecordIO",
        attribute_names=["source-ref", "annotations"],  # To guarantee only 2 attributes fed in
        shuffle_config=sagemaker.session.ShuffleConfig(seed=1337)
    )
    ```

    ...SageMaker will produce a RecordIO stream with alternating records of image and annotation.

    This generator reads batches of records from the stream and converts each into a GluonCV 
    RecordFileDetection.
    """
    ixbatch = -1
    epoch_end = False
    epoch_file = f"{channel}_{epoch}"
    epoch_records = recordio.MXRecordIO(epoch_file, "r")
    with TemporaryDirectory() as tmpdirname:
        batch_records_file = os.path.join(tmpdirname, "data.rec")
        batch_idx_file = os.path.join(tmpdirname, "data.idx")
        while not epoch_end:
            ixbatch += 1
            logger.info(f"Epoch {epoch}, stream-batch {ixbatch}, channel {channel}")

            # TODO: Wish we could use with statements for file contexts, but I think MXNet can't?
            try:
                os.remove(batch_records_file)
                os.remove(batch_idx_file)
            except OSError:
                pass
            try:
                os.mknod(batch_idx_file)
            except OSError:
                pass

            # Stream batch of data in to temporary batch_records file (pair):
            batch_records = recordio.MXIndexedRecordIO(batch_idx_file, batch_records_file, "w")
            image_raw = None
            image_meta = None
            ixdatum = 0
            invalid = False
            while (ixdatum < batch_size):
                # Read from the SageMaker stream:
                raw = epoch_records.read()
                # Determine whether this object is the image or the annotation:
                if (not raw):
                    if (image_meta or image_raw):
                        raise ValueError(
                            f"Bad stream {epoch_file}: Finished with partial record {ixdatum}...\n"
                            f"{'Had' if image_raw else 'Did not have'} image; "
                            f"{'Had' if image_raw else 'Did not have'} annotations."
                        )
                    epoch_end = True
                    break
                elif (raw[0] == b"{"[0]): # Binary in Python is weird...
                    logger.debug(f"Record {ixdatum} got metadata: {raw[:20]}...")
                    if (image_meta):
                        raise ValueError(
                            f"Bad stream {epoch_file}: Already got annotations for record {ixdatum}...\n"
                            f"Existing: {image_meta}\n"
                            f"New: {raw}"
                        )
                    else:
                        image_meta = json.loads(raw)
                else:
                    logger.debug(f"Record {ixdatum} got image: {raw[:20]}...")
                    if (image_raw):
                        raise ValueError(
                            f"Bad stream {epoch_file}: Missing annotations for record {ixdatum}...\n"
                        )
                    else:
                        image_raw = raw
                        # Since a stream-batch becomes an iterable GluonCV dataset, to which
                        # downstream transformations are applied in bulk, it's best to weed out any
                        # corrupted files here if possible rather than risk a whole mini-batch or
                        # stream-batch getting discarded:
                        try:
                            img = image.imdecode(bytearray(raw))
                            logger.debug(f"Loaded image shape {img.shape}")
                        except ValueError as e:
                            logger.exception("Failed to load image data - skipping...")
                            invalid = True
                        # TODO: Since we already parse images, try to buffer the tensors not JPG

                # If both image and annotation are collected, we're ready to pack for GluonCV:
                if (image_raw is not None and len(image_raw) and image_meta):
                    if invalid:
                        image_raw = None
                        image_meta = None
                        invalid = False
                        continue

                    if (image_meta.get("image_size")):
                        image_width = image_meta["image_size"][0]["width"]
                        image_height = image_meta["image_size"][0]["height"]
                        boxes = [[
                            ann["class_id"],
                            ann["left"] / image_width,
                            ann["top"] / image_height,
                            (ann["left"] + ann["width"]) / image_width,
                            (ann["top"] + ann["height"]) / image_height
                        ] for ann in image_meta["annotations"]]
                    else:
                        logger.debug(
                            "Writing non-normalized bounding box (no image_size in manifest)"
                        )
                        boxes = [[
                            ann["class_id"],
                            ann["left"],
                            ann["top"],
                            ann["left"] + ann["width"],
                            ann["top"] + ann["height"]
                        ] for ann in image_meta["annotations"]]

                    boxes_flat = [ val for box in boxes for val in box ]
                    header_data = [2, 5] + boxes_flat
                    logger.debug(f"Annotation header data {header_data}")
                    header = recordio.IRHeader(
                        0, # Convenience value not used
                        # Flatten nested boxes array:
                        header_data,
                        ixdatum,
                        0
                    )
                    batch_records.write_idx(ixdatum, recordio.pack(header, image_raw))
                    image_raw = None
                    image_meta = None
                    ixdatum += 1

            # Close the write stream (we'll re-open the file-pair to read):
            batch_records.close()

            if ixdatum == 0:
                logger.debug("Reached end of stream with no valid records - discarding")
                break

            if (epoch_end and discard_partial_final):
                logger.debug("Discarding final partial batch")
                break # (Don't yield the part-completed batch)

            dataset = RecordFileDetection(batch_records_file)
            logger.debug(f"Stream batch ready with {len(dataset)} records")
            if not len(dataset):
                raise ValueError(
                    "Why is the dataset empty after loading as RecordFileDetection!?!?"
                )
            yield dataset
