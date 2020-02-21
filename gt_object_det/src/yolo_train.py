"""Train YOLOv3 with random shapes."""

# Python Built-Ins:
import argparse
import json
import logging
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import warnings

# Inline Installations:
import numpy as np
subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "gluoncv"])

# External Dependencies:
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import MixupDetection
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
#from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler, LRSequential

# Local Dependencies:
# TODO: Replace with standard gluoncv.utils.metrics.voc_detection.VOC07MApMetric?
from hello import VOC07MApMetric
# Export functions for deployment in SageMaker:
from sm_gluoncv_hosting import *


logger = 1 # TODO: logging.getLogger()


def parse_args():
    hps = json.loads(os.environ["SM_HPS"])
    parser = argparse.ArgumentParser(description="Train YOLO networks with random input shape.")
    
    # Hyperparameters:
    parser.add_argument("--network", type=str, default=hps.get("network", "yolo3_darknet53_coco"),
        help="Base network name which serves as feature extraction base."
    )
    parser.add_argument("--data-shape", type=int, default=hps.get("data-shape", 320),
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
    parser.add_argument("--epochs", type=int, default=hps.get("epochs", 1),
        help="The maximum number of passes over the training data."
    )
    parser.add_argument("--start-epoch", type=int, default=hps.get("start-epoch", 0),
        help="Starting epoch for resuming, default is 0 for new training."
            "You can specify it to 100 for example to start from 100 epoch."
    )
    parser.add_argument("--resume", type=str, default=hps.get("resume", ""),
        help="Resume from previously saved parameters file, e.g. ./yolo3_xxx_0123.params. "
    )
    parser.add_argument("--optimizer", type=str, default=hps.get("optimizer", "sgd"),
        help="Optimizer used for training"
    )
    parser.add_argument("--lr", type=float, default=hps.get("lr", hps.get("learning-rate", 0.001)),
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
    parser.add_argument("--wd", type=float, default=hps.get("wd", hps.get("weight-decay", 0.0005)),
        help="Weight decay"
    )
    parser.add_argument("--no-wd", action="store_true",
        help="Whether to remove weight decay on bias, and beta/gamma for batchnorm layers."
    )
    parser.add_argument("--val-interval", type=int, default=hps.get("val-interval", 1),
        help="Epoch interval for validation, increasing will reduce the training time if validation is slow"
    )
    parser.add_argument("--seed", type=int, default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)."
    )
    parser.add_argument("--syncbn", action="store_true", default=hps.get("syncbn", False),
        help="Use synchronize BN across devices."
    )
    parser.add_argument("--mixup", action="store_true", default=hps.get("mixup", False),
        help="whether to enable mixup." # TODO: What?
    )
    parser.add_argument("--no-mixup-epochs", type=int, default=hps.get("no-mixup-epochs", 20),
        help="Disable mixup training if enabled in the last N epochs."
    )
    parser.add_argument("--pretrained", action="store_false", default=hps.get("pretrained", True),
        help="Use pretrained weights"
    )
    parser.add_argument("--label-smooth", action="store_true", default=hps.get("label-smooth", False),
        help="Use label smoothing."
    )

    # Core Parameters
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS'),
                        help='Number of GPUs to use in training.')
    parser.add_argument("--num-workers", "-j", type=int, default=hps.get("num-workers", 0),
        help='Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful'
    )
    
    # I/O Settings:
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--checkpoints", type=str, default=None)
    parser.add_argument("--train", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--validation", type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument("--stream-batch-size", type=int, default=hps.get("stream-batch-size", 12),
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
    parser.add_argument("--save-prefix", type=str, default=hps.get("save-prefix", ""),
        help="Saving parameter prefix"
    )
    parser.add_argument("--save-interval", type=int, default=hps.get("save-interval", 10),
        help="Saving parameters epoch interval, best model will always be saved."
    )
    
    args = parser.parse_args()
    return args

# TODO: Remove - No longer used if new data-loading works
# def get_dataloader(net, dataset, data_shape, batch_size, validation:bool, args):
#     """Get dataloader."""
#     width, height = data_shape, data_shape
#     if validation:
#         logger.debug("Creating validation DataLoader")
#         batchify_fn=Tuple(Stack(), Pad(pad_val=-1))
#         return gluon.data.DataLoader(
#             dataset.transform(YOLO3DefaultValTransform(width, height)),
#             batch_size,
#             shuffle=True,
#             batchify_fn=batchify_fn,
#             last_batch="keep",
#             num_workers=args.num_workers
#         )
#     else:
#         transformed_dataset = dataset.transform(YOLO3DefaultTrainTransform(width, height, mixup=args.mixup))
#         for ix in range(len(dataset)):
#             datum = dataset[ix]
#             img, label = dataset[ix]
#             tdatum = transformed_dataset[ix]
#             timg, tlabel = tdatum
#             print(f"Record {ix} has len {len(datum)}, comprising img shape {img.shape}, label shape{label.shape}. Detections:")
#             print(label)
#             print(f"Transformed detections len {len(tdatum)}, label:")
#             print(tlabel)
#         if True: #if args.no_random_shape:
#             logger.debug("Creating DataLoader without random transform")
#             batchify_fn = Tuple(Stack(), Pad(axis=0, pad_val=-1))
#             #batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
#             return gluon.data.DataLoader(
#                 dataset.transform(YOLO3DefaultTrainTransform(width, height, mixup=args.mixup)),
#                 batch_size, shuffle=True,
#                 batchify_fn=batchify_fn,
#                 last_batch="discard", num_workers=args.num_workers
#             )
#         else:
#             logger.debug("Creating DataLoader with random transform")
#             # Stack images, all targets generated:
#             batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
#             transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, mixup=args.mixup) for x in range(10, 20)]
#             return RandomTransformDataLoader(
#                 transform_fns, dataset, batch_size=batch_size, interval=10, last_batch="discard",
#                 shuffle=True, batchify_fn=batchify_fn, num_workers=args.num_workers
#             )

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data_channel, epoch, ctx, eval_metric, transforms, batchify_fn, args):
    """Test on validation dataset."""
    eval_metric.reset()
    val_data_gen = pipe_detection_minibatch(epoch, channel=val_data_channel, batch_size=args.stream_batch_size)
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    metric_updated = False
    for val_dataset in val_data_gen:
        val_dataloader = gluon.data.DataLoader(
            dataset.transform(transforms),
            args.batch_size,
            shuffle=True,
            batchify_fn=batchify_fn,
            last_batch="keep",
            num_workers=args.num_workers
        )

        # TODO: Remove if new data-loading above works
#         val_dataloader = get_dataloader(
#             net, val_dataset, args.data_shape, args.batch_size, validation=True, args=args
#         )
        for batch in val_dataloader:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
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
        raise ValueError("Validation metric was never updated by a mini-batch: Is your validation data set empty?")
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
    epoch_file = channel + f"_{epoch}"
    epoch_records = mx.recordio.MXRecordIO(epoch_file, "r")
    with TemporaryDirectory() as tmpdirname:
        batch_records_file = os.path.join(tmpdirname, "data.rec")
        batch_idx_file = os.path.join(tmpdirname, "data.idx")
        while not epoch_end:
            ixbatch += 1
            logger.info("Epoch %d, stream minibatch %d, channel %s" % (epoch, ixbatch, channel))
            
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
                        image_raw = raw # TODO: It's weird that we don't have to unpack() this?
                
                # If both image and annotation have been collected, we're ready to pack for GluonCV:
                if (image_raw and image_meta):
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
                        logger.debug("Writing non-normalized bounding box (no image_size in manifest)")
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
            kvstore='local'
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
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()

    # Intermediate Metrics:
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')
    
    # Evaluation Metrics:
    val_metric = VOC07MApMetric(iou_thresh=0.5)

    # Data transformations:
    train_batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    train_transforms = (
        YOLO3DefaultTrainTransform(args.data_shape, args.data_shape, net=async_net, mixup=args.mixup)
        if args.no_random_shape else
        [YOLO3DefaultTrainTransform(x * 32, x * 32, net=async_net, mixup=args.mixup) for x in range(10, 20)]
    )
    validation_batchify_fn = None
    validation_transforms = None
    if args.validation:
        validation_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        validation_transforms = YOLO3DefaultValTransform(args.data_shape, args.data_shape)

    logger.info(args)
    logger.info(f"Start training from [Epoch {args.start_epoch}]")
    best_map = [0]
    logger.info('Sleeping for 3s in case training data file not yet ready')
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
        train_data_gen = pipe_detection_minibatch(epoch, channel=args.train, batch_size=args.stream_batch_size)
        for ix_streambatch, train_dataset in enumerate(train_data_gen):
            # TODO: Mixup is kinda rubbish if it's only within a (potentially small) batch
            if args.mixup:
                train_dataset = MixupDetection(train_dataset)

            # Create dataloader for the stream-batch:
            if args.no_random_shape:
                logger.debug("Creating train DataLoader without random transform")
                train_dataloader = gluon.data.DataLoader(
                    train_dataset.transform(train_transforms),
                    args.batch_size, shuffle=True,
                    batchify_fn=train_batchify_fn,
                    last_batch="discard", num_workers=args.num_workers
                )
            else:
                logger.debug("Creating train DataLoader with random transform")
                train_dataloader = RandomTransformDataLoader(
                    train_transforms, train_dataset, batch_size=args.batch_size, interval=10, last_batch="discard",
                    shuffle=True, batchify_fn=train_batchify_fn, num_workers=args.num_workers
                )
            
            # TODO: Remove if above dataloading works
            #train_dataloader = get_dataloader(
            #    async_net, train_dataset, args.data_shape, args.batch_size, validation=False, args=args
            #)
            
            if args.mixup:
                logger.info("Shuffling stream-batch")
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
            
            logger.info("Training on stream-batch %d (%d records)" % (ix_streambatch, len(train_dataset)))
            for i, batch in enumerate(train_dataloader):
                logger.info("Epoch %d, stream batch %d, minibatch %d" % (epoch, ix_streambatch, i))

                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0, even_split=False) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0, even_split=False)
                sum_losses = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    autograd.backward(sum_losses)            
                trainer.step(batch_size)
                obj_metrics.update(0, obj_losses)
                center_metrics.update(0, center_losses)
                scale_metrics.update(0, scale_losses)
                cls_metrics.update(0, cls_losses)
                if args.log_interval and not (i + 1) % args.log_interval:
                    name1, loss1 = obj_metrics.get()
                    name2, loss2 = center_metrics.get()
                    name3, loss3 = scale_metrics.get()
                    name4, loss4 = cls_metrics.get()
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info(
            '[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4
            )
        )
        if not (epoch + 1) % args.val_interval:
            logger.info(f"Validating epoch {epoch + 1}")
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, args.validation, epoch, ctx, VOC07MApMetric(iou_thresh=0.5), validation_transforms, validation_batchify_fn, args)
            if isinstance(map_name, list):
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                #train_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name_train, mean_ap_train)])
                current_map = float(mean_ap[-1])
            else:
                val_msg='{}={}'.format(map_name, mean_ap)
                #train_msg='{}={}'.format(map_name_train, mean_ap_train)
                current_map = mean_ap
            logger.info('[Epoch {}] Validation: {} ;'.format(epoch, val_msg))
            #logger.info('[Epoch {}] Train: {} ;'.format(epoch, train_msg))  
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, os.path.join(args.model_dir, "yolov3"))
        

if __name__ == "__main__":
    args = parse_args()
    # Fix seed for mxnet, numpy and python builtin random generator.
    if args.seed:
        gutils.random.seed(args.seed)
    
    # Set up logger
    # TODO: What if not in training mode?
    logging.basicConfig()
    #global logger
    logger = logging.getLogger()
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)
    log_file_path = args.save_prefix + "_train.log"
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
    args.save_prefix += net_name
    # use sync bn if specified
    num_sync_bn_devices = len(ctx) if args.syncbn else -1
    #classes = read_classes(args)
    net = None
    if num_sync_bn_devices > 1:
        print("num_sync_bn_devices > 1")
        if args.pretrained:
            print("use pretrained weights of coco")
            net = get_model(net_name, pretrained=True, num_sync_bn_devices=num_sync_bn_devices)
        else:        
            print("use pretrained weights of mxnet")
            net = get_model(net_name, pretrained_base=True, num_sync_bn_devices=num_sync_bn_devices)

        #net.reset_class(classes)            
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
    else:
        print("num_sync_bn_devices <= 1")
        net = get_model(net_name, pretrained=True)
        #if args.pretrained:
        #    net = get_model(net_name, pretrained=True)            
        #else:
        #    net = get_model(net_name, pretrained_base=True)
        #net.reset_class(classes)
        async_net = net

    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
    
    # training
    train(net, async_net, ctx, args)
