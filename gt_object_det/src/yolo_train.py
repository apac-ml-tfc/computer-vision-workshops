"""Train YOLOv3 with random shapes."""

# Python Built-Ins:
import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import warnings

# Although a requirements.txt file is supported at train time, it doesn't get installed for
# inference and we need GluonCV then too... So unfortunately will have to inline install:
subprocess.call([sys.executable, "-m", "pip", "install", "gluoncv==0.6.0"])

# External Dependencies:
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import MixupDetection
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler, LRSequential
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet.ndarray.contrib import isfinite
import numpy as np

# Local Dependencies:
# Export functions for deployment in SageMaker:
from sm_gluoncv_hosting import *


logger = 1 # TODO: logging.getLogger()

def boolean_hyperparam(raw):
    """Boolean argparse type for convenience in SageMaker

    SageMaker HPO supports categorical variables, but doesn't have a specific type for booleans -
    so passing `command --flag` to our container is tricky but `command --arg true` is easy.

    Using argparse with the built-in `type=bool`, the only way to set false would be to pass an
    explicit empty string like: `command --arg ""`... which looks super weird and isn't intuitive.

    Using argparse with `type=boolean_hyperparam` instead, the CLI will support all the various
    ways to indicate 'yes' and 'no' that you might expect: e.g. `command --arg false`.

    """
    valid_false = ("0", "false", "n", "no", "")
    valid_true = ("1", "true", "y", "yes")
    raw_lower = raw.lower()
    if raw_lower in valid_false:
        return False
    elif raw_lower in valid_true:
        return True
    else:
        raise argparse.ArgumentTypeError(
        f"'{raw}' value for case-insensitive boolean hyperparam is not in valid falsy "
        f"{valid_false} or truthy {valid_true} value list"
    )

def parse_args():
    hps = json.loads(os.environ["SM_HPS"])
    parser = argparse.ArgumentParser(description="Train YOLO networks with random input shape.")
    
    # Network parameters:
    parser.add_argument("--network", type=str, default=hps.get("network", "yolo3_darknet53_coco"),
        help="Base network name which serves as feature extraction base."
    )
    parser.add_argument("--pretrained", type=boolean_hyperparam,
        default=hps.get("pretrained", True),
        help="Use pretrained weights"
    )
    parser.add_argument("--num-classes", type=int, default=hps.get("num-classes", 1),
        help="Number of classes in training data set."
    )
    parser.add_argument("--data-shape", type=int, default=hps.get("data-shape", 416),
        help="Input data shape for evaluation, use 320, 416, 608... "
            "Training is with random shapes from (320 to 608)."
    )
    parser.add_argument("--no-random-shape", action="store_true",
        help="Use fixed size(data-shape) throughout the training, which will be faster "
            "and require less memory. However, final model will be slightly worse."
    )
    parser.add_argument("--batch-size", type=int, default=hps.get("batch-size", 4),
        help="Training mini-batch size"
    )

    # Training process parameters:
    parser.add_argument("--epochs", type=int, default=hps.get("epochs", 1),
        help="The maximum number of passes over the training data."
    )
    parser.add_argument("--start-epoch", type=int, default=hps.get("start-epoch", 0),
        help="Starting epoch for resuming, default is 0 for new training."
            "You can specify it to 100 for example to start from 100 epoch."
    )
    parser.add_argument("--resume", type=str, default=hps.get("resume", ""),
        help="Resume from previously saved parameters file, e.g. ./yolo3_xxx_0123.params"
    )
    parser.add_argument("--optimizer", type=str, default=hps.get("optimizer", "sgd"),
        help="Optimizer used for training"
    )
    parser.add_argument("--lr", "--learning-rate", type=float,
        default=hps.get("lr", hps.get("learning-rate", 0.0001)),
        help="Learning rate"
    )
    parser.add_argument("--lr-mode", type=str, default=hps.get("lr-mode", "step"),
        help="Learning rate scheduler mode. Valid options are step, poly and cosine."
    )
    parser.add_argument("--lr-decay", type=float, default=hps.get("lr-decay", 0.1),
        help="Decay rate of learning rate. default is 0.1."
    )
    parser.add_argument("--lr-decay-period", type=int, default=hps.get("lr-decay-period", 0),
        help="Interval for periodic learning rate decays, or 0 to disable."
    )
    parser.add_argument("--lr-decay-epoch", type=str, default=hps.get("lr-decay-epoch", "160,180"),
        help="Epochs at which learning rate decays."
    )
    parser.add_argument("--warmup-lr", type=float, default=hps.get("warmup-lr", 0.0),
        help="Starting warmup learning rate."
    )
    parser.add_argument("--warmup-epochs", type=int, default=hps.get("warmup-epochs", 0),
        help="Number of warmup epochs."
    )
    parser.add_argument("--momentum", type=float, default=hps.get("momentum", 0.9),
        help="SGD momentum"
    )
    parser.add_argument("--wd", "--weight-decay", type=float,
        default=hps.get("wd", hps.get("weight-decay", 0.0005)),
        help="Weight decay"
    )
    parser.add_argument("--no-wd", action="store_true",
        help="Whether to remove weight decay on bias, and beta/gamma for batchnorm layers."
    )
    parser.add_argument("--val-interval", type=int, default=hps.get("val-interval", 1),
        help="Epoch interval for validation, raise to reduce training time if validation is slow"
    )
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)."
    )
    parser.add_argument("--mixup", type=boolean_hyperparam, default=hps.get("mixup", False),
        help="whether to enable mixup." # TODO: What?
    )
    parser.add_argument("--no-mixup-epochs", type=int, default=hps.get("no-mixup-epochs", 20),
        help="Disable mixup training if enabled in the last N epochs."
    )
    parser.add_argument("--label-smooth", type=boolean_hyperparam,
        default=hps.get("label-smooth", False),
        help="Use label smoothing."
    )
    parser.add_argument("--early-stopping", type=boolean_hyperparam,
        default=hps.get("early-stopping", False),
        help="Enable early stopping."
    )
    parser.add_argument("--early-stopping-min-epochs", type=int,
        default=hps.get("early-stopping-min-epochs", 20),
        help="Minimum number of epochs to train before allowing early stop."
    )
    parser.add_argument("--early-stopping-patience", type=int,
        default=hps.get("early-stopping-patience", 5),
        help="Maximum number of epochs to wait for a decreased loss before stopping early."
    )

    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 0),
        help="Number of GPUs to use in training."
    )
    parser.add_argument("--num-workers", "-j", type=int,
        default=hps.get("num-workers", max(0, int(os.environ.get("SM_NUM_CPUS", 0)) - 2)),
        help='Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful'
    )
    parser.add_argument("--syncbn", type=boolean_hyperparam, default=hps.get("syncbn", False),
        help="Use synchronize BN across devices."
    )

    # I/O Settings:
    parser.add_argument("--output-data-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--checkpoint-dir", type=str,
        default=hps.get("checkpoint-dir", "/opt/ml/checkpoints")
    )
    parser.add_argument("--checkpoint-interval", type=int,
        default=hps.get("checkpoint-interval", 0),
        help="Epochs between saving checkpoints (set 0 to disable)"
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--stream-batch-size", type=int, default=hps.get("stream-batch-size", 16),
        help="S3 data streaming batch size (for good randomization, set >> batch-size)"
    )
    parser.add_argument("--log-interval", type=int, default=hps.get("log-interval", 100),
        help="Logging mini-batch interval. Default is 100."
    )
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )
    parser.add_argument("--num-samples", type=int, default=hps.get("num-samples", -1),
        help="(Limit) number of training images, or -1 to take all automatically."
    )
    parser.add_argument("--save-interval", type=int, default=hps.get("save-interval", 10),
        help="Saving parameters epoch interval, best model will always be saved."
    )

    args = parser.parse_args()

    # Post-argparse validations:
    args.resume = args.resume.strip()
    return args

def save_progress(
    net,
    current_score,
    prev_best_score,
    best_folder,
    epoch,
    checkpoint_interval,
    checkpoints_folder,
    model_prefix="model",
):
    """Save checkpoints if appropriate, and best model if current_score > prev_best_score
    """
    current_score = float(current_score)
    if current_score > prev_best_score:
        # HybridBlock.export() saves path-symbol.json and path-####.params (4-padded epoch number)
        os.makedirs(best_folder, exist_ok=True)
        net.export(os.path.join(best_folder, model_prefix), epoch)
        logger.info(f"New best model at epoch {epoch}: {current_score} over {prev_best_score}")

        # Avoid cluttering up the best_folder with extra params:
        # We do this after export()ing even though it makes things more complex, in case an export
        # error caused us to first delete our old model, then fail to replace it!
        for f in glob.glob(f"{os.path.join(best_folder, model_prefix)}-*.params"):
            if int(f.rpartition(".")[0].rpartition("-")[2]) < epoch:
                logger.debug(f"Deleting old file {f}")
                os.remove(f)

        if checkpoints_folder and checkpoint_interval:
            os.makedirs(os.path.join(args.checkpoint_dir, "best"), exist_ok=True)
            shutil.copy(
                os.path.join(best_folder, f"{model_prefix}-symbol.json"),
                os.path.join(checkpoints_folder, "best", f"{model_prefix}-symbol.json")
            )
            shutil.copy(
                os.path.join(best_folder, f"{model_prefix}-{epoch:04d}.params"),
                os.path.join(checkpoints_folder, "best", f"{model_prefix}-best.params")
            )
    if checkpoints_folder and checkpoint_interval and (epoch % checkpoint_interval == 0):
        os.makedirs(os.path.join(args.checkpoint_dir, f"{epoch:04d}"), exist_ok=True)
        net.export(os.path.join(checkpoints_folder, f"{epoch:04d}", model_prefix), epoch)


def validate(net, val_data_channel, epoch, ctx, eval_metric, transforms, batchify_fn, args):
    """Test on validation dataset."""
    eval_metric.reset()
    val_data_gen = pipe_detection_minibatch(
        epoch,
        channel=val_data_channel,
        batch_size=args.stream_batch_size
    )
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    metric_updated = False
    for val_dataset in val_data_gen:
        val_dataloader = gluon.data.DataLoader(
            val_dataset.transform(transforms),
            args.batch_size,
            shuffle=True,
            batchify_fn=batchify_fn,
            last_batch="keep",
            num_workers=args.num_workers
        )

        for batch in val_dataloader:
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0, even_split=False
            )
            label = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0, even_split=False
            )
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []        
            for x, y in zip(data, label):
                print(".", end="")
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))

            # update metric        
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)
            metric_updated = True

    if not metric_updated:
        raise ValueError(
            "Validation metric was never updated by a mini-batch: "
            "Is your validation data set empty?"
        )
    return eval_metric.get()


def pipe_detection_minibatch(
    epoch:int,
    batch_size:int=50,
    channel:str="/opt/ml/input/data/train",
    discard_partial_final:bool=False
):
    """Generator for batched GluonCV RecordFileDetectors from SageMaker Pipe Mode stream

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
    epoch_records = mx.recordio.MXRecordIO(epoch_file, "r")
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
            batch_records = mx.recordio.MXIndexedRecordIO(batch_idx_file, batch_records_file, "w")
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
                            img = mx.image.imdecode(bytearray(raw))
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
                    header = mx.recordio.IRHeader(
                        0, # Convenience value not used
                        # Flatten nested boxes array:
                        header_data,
                        ixdatum,
                        0
                    )
                    batch_records.write_idx(ixdatum, mx.recordio.pack(header, image_raw))
                    image_raw = None
                    image_meta = None
                    ixdatum += 1

            # Close the write stream (we'll re-open the file-pair to read):
            batch_records.close()

            if (epoch_end and discard_partial_final):
                logger.debug("Discarding final partial batch")
                break # (Don't yield the part-completed batch)

            dataset = gcv.data.RecordFileDetection(batch_records_file)
            logger.debug(f"Stream batch ready with {len(dataset)} records")
            if not len(dataset):
                raise ValueError(
                    "Why is the dataset empty after loading as RecordFileDetection!?!?"
                )
            yield dataset


def train(net, async_net, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params(".*beta|.*gamma|.*bias").items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

    lr_scheduler = LRSequential([
        LRScheduler("linear", base_lr=0, target_lr=args.lr,
                    nepochs=args.warmup_epochs, iters_per_epoch=args.batch_size),
        LRScheduler(args.lr_mode, base_lr=args.lr,
                    nepochs=args.epochs - args.warmup_epochs,
                    iters_per_epoch=args.batch_size,
                    step_epoch=lr_decay_epoch,
                    step_factor=args.lr_decay, power=2),
    ])
    if (args.optimizer == "sgd"):
        trainer = gluon.Trainer(
            net.collect_params(),
            args.optimizer,
            { "wd": args.wd, "momentum": args.momentum, "lr_scheduler": lr_scheduler },
            kvstore="local"
        )
    elif (args.optimizer == "adam"):
        trainer = gluon.Trainer(
            net.collect_params(),
            args.optimizer,
            { "lr_scheduler": lr_scheduler },
            kvstore="local"
        )
    else:
        trainer = gluon.Trainer(net.collect_params(), args.optimizer, kvstore="local")

    # targets
    #sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    #l1_loss = gluon.loss.L1Loss()

    # Intermediate Metrics:
    train_metrics = (
        mx.metric.Loss("ObjLoss"),
        mx.metric.Loss("BoxCenterLoss"),
        mx.metric.Loss("BoxScaleLoss"),
        mx.metric.Loss("ClassLoss"),
        mx.metric.Loss("TotalLoss"),
    )
    train_metric_ixs = range(len(train_metrics))
    target_metric_ix = -1  # Train towards TotalLoss (the last one)

    # Evaluation Metrics:
    val_metric = VOC07MApMetric(iou_thresh=0.5)

    # Data transformations:
    train_batchify_fn = Tuple(
        *([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)])
    )
    train_transforms = (
        YOLO3DefaultTrainTransform(
            args.data_shape,
            args.data_shape,
            net=async_net,
            mixup=args.mixup
        )
        if args.no_random_shape else
        [
            YOLO3DefaultTrainTransform(x * 32, x * 32, net=async_net, mixup=args.mixup)
            for x in range(10, 20)
        ]
    )
    validation_batchify_fn = None
    validation_transforms = None
    if args.validation:
        validation_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        validation_transforms = YOLO3DefaultValTransform(args.data_shape, args.data_shape)

    logger.info(args)
    logger.info(f"Start training from [Epoch {args.start_epoch}]")
    prev_best_score = float("-inf")
    best_epoch = args.start_epoch
    logger.info("Sleeping for 3s in case training data file not yet ready")
    time.sleep(3)
    for epoch in range(args.start_epoch, args.epochs):
#         if args.mixup:
#             # TODO(zhreshold): more elegant way to control mixup during runtime
#             try:
#                 train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
#             except AttributeError:
#                 train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
#             if epoch >= args.epochs - args.no_mixup_epochs:
#                 try:
#                     train_data._dataset.set_mixup(None)
#                 except AttributeError:
#                     train_data._dataset._data.set_mixup(None)

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()

        logger.debug(f'Input data dir contents: {os.listdir("/opt/ml/input/data/")}')
        train_data_gen = pipe_detection_minibatch(
            epoch, channel=args.train, batch_size=args.stream_batch_size
        )
        for ix_streambatch, train_dataset in enumerate(train_data_gen):
            # TODO: Mixup is kinda rubbish if it's only within a (potentially small) batch
            if args.mixup:
                train_dataset = MixupDetection(train_dataset)

            # Create dataloader for the stream-batch:
            if args.no_random_shape:
                logger.debug("Creating train DataLoader without random transform")
                train_dataloader = gluon.data.DataLoader(
                    train_dataset.transform(train_transforms),
                    batch_size=args.batch_size,
                    batchify_fn=train_batchify_fn,
                    last_batch="discard",
                    num_workers=args.num_workers,
                    shuffle=True,
                )
            else:
                logger.debug("Creating train DataLoader with random transform")
                train_dataloader = RandomTransformDataLoader(
                    train_transforms,
                    train_dataset,
                    interval=10,
                    batch_size=args.batch_size,
                    batchify_fn=train_batchify_fn,
                    last_batch="discard",
                    num_workers=args.num_workers,
                    shuffle=True,
                )

            if args.mixup:
                logger.debug("Shuffling stream-batch")
                # TODO(zhreshold): more elegant way to control mixup during runtime
                try:
                    train_dataloader._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    train_dataloader._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= args.epochs - args.no_mixup_epochs:
                    try:
                        train_dataloader._dataset.set_mixup(None)
                    except AttributeError:
                        train_dataloader._dataset._data.set_mixup(None)

            logger.debug(
                f"Training on stream-batch {ix_streambatch} ({len(train_dataset)} records)"
            )
            # TODO: Improve stream-batching robustness to drop loop guard clauses
            # While it would be nice to simply `for i, batch in enumerate(train_dataloader):`,
            # corrupted image buffers are somehow sneaking through the stream-batch at the moment.
            #
            # For now, we catch and tolerate these errors - trying to resume stream-batch process
            # where possible and otherwise discarding the remainder of the stream-batch :-(
            done = False
            i = -1
            dataiter = iter(train_dataloader)
            while not done:
                i += 1
                batch = None
                while not batch:
                    try:
                        batch = next(dataiter)
                    except StopIteration:
                        done = True
                        break
                    except ValueError:
                        # Some problem with the minibatch prevented loading - try the next
                        logger.warn(
                            f"[Epoch {epoch}][Streambatch {ix_streambatch}] "
                            f"Failed to load minibatch {i}, trying next..."
                        )
                        i += 1
                    except:
                        logger.error(
                            f"[Epoch {epoch}][Streambatch {ix_streambatch}] "
                            f"Failed to iterate minibatch {i}: Discarding remainder"
                        )
                        break

                if not batch:
                    logger.debug(
                        f"[Epoch {epoch}][Streambatch {ix_streambatch}] "
                        f"Done after {i} minibatches"
                    )
                    break
                logger.debug(f"Epoch {epoch}, stream batch {ix_streambatch}, minibatch {i}")

                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [
                    gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0, even_split=False)
                    for it in range(1, 6)
                ]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0, even_split=False)
                loss_trackers = tuple([] for metric in train_metrics)
                with autograd.record():
                    for ix, x in enumerate(data):
                        losses_raw = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        # net outputs: [obj_loss, center_loss, scale_loss, cls_loss]
                        # Each a mx.ndarray 1xbatch_size. This is the same order as our
                        # train_metrics, so we just need to add a total vector:
                        total_loss = sum(losses_raw)
                        losses = losses_raw + [total_loss]

                        # If any sample's total loss is non-finite, sum will be:
                        if not isfinite(sum(total_loss)):
                            logger.error(
                                f"[Epoch {epoch}][Streambatch {ix_streambatch}][Minibatch {i}] "
                                f"got non-finite losses: {losses_raw}")
                            # TODO: Terminate training if losses or gradient go infinite?

                        for ix in train_metric_ixs:
                            loss_trackers[ix].append(losses[ix])

                    autograd.backward(loss_trackers[target_metric_ix])
                trainer.step(batch_size)
                for ix in train_metric_ixs:
                    train_metrics[ix].update(0, loss_trackers[ix])

                if args.log_interval and not (i + 1) % args.log_interval:
                    train_metrics_current = map(lambda metric: metric.get(), train_metrics)
                    metrics_msg = "; ".join(
                        [f"{name}={val:.3f}" for name, val in train_metrics_current]
                    )
                    logger.info(
                        f"[Epoch {epoch}][Streambatch {ix_streambatch}][Minibatch {i}] "
                        f"LR={trainer.learning_rate:.2E}; "
                        f"Speed={batch_size/(time.time()-btic):.3f} samples/sec; {metrics_msg};"
                    )
                btic = time.time()

        train_metrics_current = map(lambda metric: metric.get(), train_metrics)
        metrics_msg = "; ".join([f"{name}={val:.3f}" for name, val in train_metrics_current])
        logger.info(f"[Epoch {epoch}] TrainingCost={time.time()-tic:.3f}; {metrics_msg};")

        if not (epoch + 1) % args.val_interval:
            logger.info(f"Validating [Epoch {epoch}]")

            metric_names, metric_values = validate(
                net, args.validation, epoch, ctx, VOC07MApMetric(iou_thresh=0.5),
                validation_transforms, validation_batchify_fn, args
            )
            if isinstance(metric_names, list):
                val_msg = "; ".join([f"{k}={v}" for k, v in zip(metric_names, metric_values)])
                current_score = float(metric_values[-1])
            else:
                val_msg = f"{metric_names}={metric_values}"
                current_score = metric_values
            logger.info(f"[Epoch {epoch}] Validation: {val_msg};")
        else:
            current_score = float("-inf")

        save_progress(
            net, current_score, prev_best_score, args.model_dir, epoch, args.checkpoint_interval,
            args.checkpoint_dir
        )
        if current_score > prev_best_score:
            prev_best_score = current_score
            best_epoch = epoch

        if (
            args.early_stopping
            and epoch >= args.early_stopping_min_epochs
            and (epoch - best_epoch) >= args.early_stopping_patience
        ):
            logger.info(
                f"[Epoch {epoch}] No improvement since epoch {best_epoch}: Stopping early"
            )
            break


if __name__ == "__main__":
    args = parse_args()

    # Fix seed for mxnet, numpy and python builtin random generator.
    if args.seed:
        gutils.random.seed(args.seed)

    # Set up logger
    # TODO: What if not in training mode?
    logging.basicConfig()
    logger = logging.getLogger()
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)
    log_file_path = args.output_data_dir + "train.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in range(args.num_gpus)]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = args.network
    # use sync bn if specified
    num_sync_bn_devices = len(ctx) if args.syncbn else -1

    logger.info(f"num_sync_bn_devices = {num_sync_bn_devices}")
    # TODO: Fix num_sync_bn_devices in darknet
    # Currently TypeError: __init__() got an unexpected keyword argument 'num_sync_bn_devices'
    # File "/usr/local/lib/python3.6/site-packages/gluoncv/model_zoo/yolo/darknet.py", line 81, in __init__
    #    super(DarknetV3, self).__init__(**kwargs)
    if args.pretrained:
        logger.info("Use pretrained weights of COCO")
        if num_sync_bn_devices >= 2:
            net = get_model(net_name, pretrained=True, num_sync_bn_devices=num_sync_bn_devices)
        else:
            net = get_model(net_name, pretrained=True)
    else:
        logger.info("Use pretrained weights of MXNet")
        if num_sync_bn_devices >= 2:
            net = get_model(net_name, pretrained_base=True, num_sync_bn_devices=num_sync_bn_devices)
        else:
            net = get_model(net_name, pretrained_base=True)

    net.reset_class(range(args.num_classes))

    # Async net used by CPU worker (if applicable):
    async_net = get_model(net_name, pretrained_base=False) if num_sync_bn_devices > 1 else net

    if args.resume:
        net.load_parameters(args.resume)
        async_net.load_parameters(args.resume)
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()

    # training
    train(net, async_net, ctx, args)
