"""SageMaker SDK-compatible Deserializers for Semantic Segmentation algorithm response types"""

# External Dependencies
from numpy import array, reshape, squeeze
from PIL import Image
from sagemaker.amazon.record_pb2 import Record
from sagemaker.deserializers import BaseDeserializer


class ImageDeserializer(BaseDeserializer):
    """Deserialize a PIL-compatible stream of Image bytes into a numpy pixel array"""
    def __init__(self, accept="image/png"):
        self.accept = accept

    @property
    def ACCEPT(self):
        return (self.accept,)

    def deserialize(self, stream, content_type):
        """Read a stream of image bytes returned from an inference endpoint.
        Args:
            stream (botocore.response.StreamingBody): A stream of bytes.
            content_type (str): The MIME type of the data.
        Returns:
            array: numpy pixel array of returned image
        """
        try:
            return array(Image.open(stream))
        finally:
            stream.close()

class HackyProtobufDeserializer(BaseDeserializer):
    """Deserialize recordio-protobuf SageMaker Semantic Segmentation algorithm response into a numpy array

    Because of struggling to find lightweight (AWS Lambda-friendly) libraries for reading RecordIO streams,
    this class translates logic from DMLC-Core's C++ RecordIO implementation (the one used by MXNet) into
    Python: Not a particularly performant or robust strategy, but it seems to work.
    """
    def __init__(self, accept="application/x-protobuf"):
        self.accept = accept

    @property
    def ACCEPT(self):
        return (self.accept,)

    @classmethod
    def next_recordio_record(cls, stream):
        """(The hacky part) Yield next RecordIO record from stream (or None if stream is finished)

        Find the reference DMLC C++ implementation this function tries to emulate at:
        https://github.com/dmlc/dmlc-core/blob/master/include/dmlc/recordio.h
        https://github.com/dmlc/dmlc-core/blob/master/src/recordio.cc
        """
        size = 0
        while (True):
            header = stream.read(8)  # Magic uint32 constant, plus uint32 record length indicator
            if len(header) == 0:
                return
            elif len(header) != 8 or int.from_bytes(header[:4], byteorder='little') != 0xced7230a:
                raise ValueError("Invalid RecordIO stream")
            header2 = int.from_bytes(header[4:], byteorder='little', signed=False)
            cflag = (header2 >> 29) & 7
            length = header2 & ((1 << 29) - 1)
            upper_align = ((length + 3) >> 2) << 2
            size += length
            if (cflag in (0, 3)):
                break
        print(f"Got size {size}, upper_align {upper_align}")
        # I *think* this is how upper_align is meant to work, but it's a bit of a guess...
        data = stream.read(upper_align)
        return data[:size]


    def deserialize(self, stream, content_type):
        """Read a recordio-protobuf stream from a SageMaker Semantic Segmentation algorithm endpoint
        Args:
            stream (botocore.response.StreamingBody): A stream of bytes.
            content_type (str): The MIME type of the data.
        Returns:
            array: numpy array of class probabilities per pixel
        """
        try:
            # Unpack the RecordIO wrapper first:
            reccontent = HackyProtobufDeserializer.next_recordio_record(stream)
            # Then load the protocol buffer:
            rec = Record()
            print("Parsing protobuf...")
            protobuf = rec.ParseFromString(reccontent)
            # Then read the two provided tensors `target` (predictions) and `shape`, squeeze out any batch
            # dimensions (since we'll always be predicting on a single image) and shape target appropriately:
            print("Fetching Tensors...")
            values = list(rec.features["target"].float32_tensor.values)
            shape = list(rec.features["shape"].int32_tensor.values)
            print("reshaping arrays...")
            shape = squeeze(shape)
            mask = reshape(array(values), shape)
            return squeeze(mask, axis=0)
        finally:
            stream.close()
