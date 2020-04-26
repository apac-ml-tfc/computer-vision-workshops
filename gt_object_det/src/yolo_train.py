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
import time
import warnings

# Although a requirements.txt file is supported at train time, it doesn't get installed for
# inference and we need GluonCV then too... So unfortunately will have to inline install:
subprocess.call([sys.executable, "-m", "pip", "install", "gluoncv==0.6.0"])

# External Dependencies:
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify, MixupDetection
#from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils import LRScheduler, LRSequential
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet.ndarray.contrib import isfinite
import numpy as np

# Local Dependencies:
# Export functions for deployment in SageMaker:
from sm_gluoncv_hosting import *
import config
import gluon_pipe_mode


logger = 1 # TODO: logging.getLogger()


def save_progress(
    net,
    inference_config,
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
        inference_config.save(os.path.join(best_folder, f"{model_prefix}-inference-config.json"))
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
            shutil.copy(
                os.path.join(best_folder, f"{model_prefix}-inference-config.json"),
                os.path.join(checkpoints_folder, "best", f"{model_prefix}-inference-config.json")
            )
    if checkpoints_folder and checkpoint_interval and (epoch % checkpoint_interval == 0):
        checkpoint_folder = os.path.join(checkpoints_folder, f"{epoch:04d}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        net.export(os.path.join(checkpoint_folder, f"{epoch:04d}", model_prefix), epoch)
        inference_config.save(os.path.join(checkpoint_folder, f"{model_prefix}-inference-config.json"))


def validate(net, val_dataloader, epoch, ctx, eval_metric, args):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    metric_updated = False

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
    train_dataset = gluon_pipe_mode.AugmentedManifestDetection(
        args.train,
        length=args.num_samples_train,
    )
    train_batchify_fn = batchify.Tuple(
        *(
            [batchify.Stack() for _ in range(6)]
            + [batchify.Pad(axis=0, pad_val=-1) for _ in range(1)]
        )
    )
    if args.no_random_shape:
        logger.debug("Creating train DataLoader without random transform")
        train_transforms = YOLO3DefaultTrainTransform(
            args.data_shape,
            args.data_shape,
            net=async_net,
            mixup=args.mixup
        )
        train_dataloader = gluon.data.DataLoader(
            train_dataset.transform(train_transforms),
            batch_size=args.batch_size,
            batchify_fn=train_batchify_fn,
            last_batch="discard",
            num_workers=args.num_workers,
            shuffle=False,  # Note that shuffle *cannot* be used with AugmentedManifestDetection
        )
    else:
        logger.debug("Creating train DataLoader with random transform")
        train_transforms = [
            YOLO3DefaultTrainTransform(x * 32, x * 32, net=async_net, mixup=args.mixup)
            for x in range(10, 20)
        ]
        train_dataloader = RandomTransformDataLoader(
            train_transforms,
            train_dataset,
            interval=10,
            batch_size=args.batch_size,
            batchify_fn=train_batchify_fn,
            last_batch="discard",
            num_workers=args.num_workers,
            shuffle=False,  # Note that shuffle *cannot* be used with AugmentedManifestDetection
        )
    validation_dataset = None
    validation_dataloader = None
    if args.validation:
        validation_dataset = gluon_pipe_mode.AugmentedManifestDetection(
            args.validation,
            length=args.num_samples_validation,
        )
        validation_dataloader = gluon.data.DataLoader(
            validation_dataset.transform(
                YOLO3DefaultValTransform(args.data_shape, args.data_shape),
            ),
            args.batch_size,
            shuffle=False,
            batchify_fn=batchify.Tuple(batchify.Stack(), batchify.Pad(pad_val=-1)),
            last_batch="keep",
            num_workers=args.num_workers,
        )

    # Prepare the inference-time configuration for our model's setup:
    # (This will be saved alongside our network structure/params)
    inference_config = config.InferenceConfig(image_size=args.data_shape)

    logger.info(args)
    logger.info(f"Start training from [Epoch {args.start_epoch}]")
    prev_best_score = float("-inf")
    best_epoch = args.start_epoch
    logger.info("Sleeping for 3s in case training data file not yet ready")
    time.sleep(3)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
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

        logger.debug(f"Input data dir contents: {os.listdir('/opt/ml/input/data/')}")
        for i, batch in enumerate(train_dataloader):
            logger.debug(f"Epoch {epoch}, minibatch {i}")

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
                            f"[Epoch {epoch}][Minibatch {i}] got non-finite losses: {losses_raw}"
                        )
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
                    f"[Epoch {epoch}][Minibatch {i}] LR={trainer.learning_rate:.2E}; "
                    f"Speed={batch_size/(time.time()-btic):.3f} samples/sec; {metrics_msg};"
                )
            btic = time.time()

        train_metrics_current = map(lambda metric: metric.get(), train_metrics)
        metrics_msg = "; ".join([f"{name}={val:.3f}" for name, val in train_metrics_current])
        logger.info(f"[Epoch {epoch}] TrainingCost={time.time()-tic:.3f}; {metrics_msg};")

        if not (epoch + 1) % args.val_interval:
            logger.info(f"Validating [Epoch {epoch}]")

            metric_names, metric_values = validate(
                net, validation_dataloader, epoch, ctx, VOC07MApMetric(iou_thresh=0.5), args
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
            net, inference_config,
            current_score, prev_best_score,
            args.model_dir, epoch, args.checkpoint_interval, args.checkpoint_dir,
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
    args = config.parse_args()

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
