"""SageMaker YOLOv3 training script (tf.keras in Pipe Mode)"""

# Python Built-Ins:
import argparse
from distutils.dir_util import copy_tree
import json
import logging
import os
import random
import subprocess
import sys
import tempfile

# Inline installs:
# Matplotlib is used in the image augmentations (RGB/HSV conversions):
#subprocess.call([sys.executable, "-m", "pip", "install", "matplotlib"])

# External Dependencies:
import numpy as np
from sagemaker_tensorflow import PipeModeDataset
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Local Dependencies:
import config
import data
from yolo3.convert import load_darknet_as_keras
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.util import get_random_data


assert tf.__version__.rpartition(".")[0] == "1.12", (
    f"This script targets TensorFlow v1.12... got {tf.__version__}"
)
# Most of the code here should actually be OK up to 1.15, but check sagemaker_tensorflow's supported TF
# versions before trying to upgrade.
#
# 1.15 has lots of warnings, so you might need to start filtering e.g. the below:
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


logger = logging.getLogger()

def clear_session_and_reseed(seed=None):
    """Otherwise K.clear_session() seems to de-seed random number generators..."""
    K.clear_session()
    if seed:
        # Python core:
        random.seed(args.seed)
        # Numpy:
        np.random.seed(args.seed)
        # TensorFlow:
        tf.set_random_seed(args.seed)

def get_anchors(anchors_path):
    """Load the YOLO anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, max_boxes=20, load_pretrained=False, freeze_body=2):
    """Create the inference model and wrap the training model around it"""
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    
    # The core of the model is the YOLO body:
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print(f"Created YOLOv3 model with {num_anchors} anchors and {num_classes} classes.")

    # TODO: tf.constantify expressions to fix the output layer below. For now, just output raw body:
    inference_model = model_body
    # Our inference model adds a Lambda layer to convert the output into readily interpretable boxes:
#     boxes, scores, classes = yolo_eval_layer(
#         model_body.output,
#         anchors,
#         num_classes,
#         input_shape,
#         max_boxes=max_boxes,
#         score_threshold=0.05,
#         iou_threshold=.5
#     )
#     inference_model = Model(image_input, [boxes, scores, classes])
    
    # The training model takes additional inputs: the ground truth YOLO output for the image:
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    if load_pretrained:
        # tf.keras.Model.load_weights doesn't support skip_mismatch
        model_body.load_weights(load_pretrained, by_name=True)
        print(f"Load weights {load_pretrained}")
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print(f"Freeze the first {num} layers of total {len(model_body.layers)} layers.")

    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss",
        arguments={ "anchors": anchors, "num_classes": num_classes, "ignore_thresh": 0.5 })(
        [*model_body.output, *y_true])
    train_model = Model([model_body.input, *y_true], model_loss)
    
    return train_model, inference_model
    

def train(args):
    """Train a model and save inference artifacts to args.model_path
    
    Parameters
    ----------
    args : argparse.Namespace
        See config.py for the parameter definitions
    """
    # Clear TF session and seed random number generators:
    clear_session_and_reseed(args.seed)

    anchors = get_anchors("model_data/yolo_anchors.txt")
    
    pretrain = None
    if (args.darknet):
        tmpmodel = load_darknet_as_keras(args.darknet + ".cfg", args.darknet + ".weights")
        tmpmodel.save(args.darknet + ".h5")
        pretrain = args.darknet + ".h5"
    
    train_model, inference_model = create_model(
        (args.data_shape, args.data_shape),
        anchors,
        args.num_classes,
        freeze_body=2,
        load_pretrained=pretrain,
    )

    ## Keras callbacks:
    checkpoint = (
        ModelCheckpoint(
            args.checkpoint_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            period=args.checkpoint_interval,
        )
        if args.checkpoint_interval else None
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)
    pretrain_callbacks = [checkpoint] if checkpoint else []
    train_callbacks = ([checkpoint] if checkpoint else []) + [reduce_lr, early_stopping]
    
    ## Datasets:
    # For more info on this exact batch size requirement, see:
    # https://github.com/aws/sagemaker-tensorflow-extensions/issues/46
    assert args.num_samples_train % args.batch_size == 0, (
        f"Training sample count {args.num_samples_train} is not a multiple of batch size {args.batch_size}, "
        "which can cause deadlocks with sagemaker_tensorflow.PipeModeDataset. Please preprocess to fix."
    )
    # An Augmented Manifest File channel streams one object per attribute of each JSON line. We assume just 2
    # attributes per object: source-ref (the image) and the label (the annotation) - so this pipeline parses
    # batches of 2 records (image, label) before assembling batch_size batches for final pre-processing.
    ds_train = PipeModeDataset(channel="train") \
        .repeat(args.epochs) \
        .batch(2) \
        .map(data.get_tf_parse_mapper((args.data_shape, args.data_shape), randomize=True)) \
        .batch(args.batch_size, drop_remainder=True) \
        .map(data.get_tf_train_batch_mapper(
            args.batch_size,
            (args.data_shape, args.data_shape),
            anchors, args.num_classes
        ))
    
    assert args.num_samples_train % args.batch_size == 0, (
        f"Training sample count {args.num_samples_train} is not a multiple of batch size {args.batch_size}, "
        "which can cause deadlocks with sagemaker_tensorflow.PipeModeDataset. Please preprocess to fix."
    )
    ds_val = PipeModeDataset(channel="validation") \
        .repeat(args.epochs) \
        .batch(2) \
        .map(data.get_tf_parse_mapper((args.data_shape, args.data_shape), randomize=False)) \
        .batch(args.batch_size, drop_remainder=True) \
        .map(data.get_tf_train_batch_mapper(
            args.batch_size,
            (args.data_shape, args.data_shape),
            anchors, args.num_classes
        ))

    ## Initial stabilization training
    # (If loading in pretrained parameters, train with frozen layers first to get a stable loss)
    if args.epochs_stabilize:
        logger.info(f"Pre-training for {args.epochs_stabilize} epochs...")
        train_model.compile(
            optimizer=Adam(lr=args.lr_pretrain),
            loss={
                # We calculate loss within the training "model" itself, so Keras loss = y_pred:
                "yolo_loss": lambda y_true, y_pred: y_pred
            }
        )

        train_model.fit(
            ds_train,
            epochs=args.epochs_stabilize,
            initial_epoch=0,
            callbacks=pretrain_callbacks,
            shuffle=False,
            steps_per_epoch=args.num_samples_train // args.batch_size,
            validation_data=ds_val,
            validation_steps=args.num_samples_validation // args.batch_size,
            verbose=2,
        )
    
    ## Main tuning (unfreezing all layers)
    remaining_epochs = args.epochs - args.epochs_stabilize
    if remaining_epochs > 0:
        logger.info("Unfreezing layers for remaining epochs...")
        for i in range(len(train_model.layers)):
            train_model.layers[i].trainable = True
        # Need to re-compile to apply the change:
        train_model.compile(
            optimizer=Adam(lr=args.lr),
            loss={ "yolo_loss": lambda y_true, y_pred: y_pred }
        )

        train_model.fit(
            ds_train,
            callbacks=train_callbacks,
            epochs=args.epochs,
            initial_epoch=args.epochs_stabilize,
            shuffle=False,
            steps_per_epoch=args.num_samples_train // args.batch_size,
            validation_data=ds_val,
            validation_steps=args.num_samples_validation // args.batch_size,
            verbose=2,
        )
    
    ## Save the inference model it TFServing format:
    # (In TFv2, TFServing can open Keras models automatically - but in v1 we need to save as TF model)
    #
    # We can't tf.saved_model.simple_save quite yet, because our TF session is full of training nodes like
    # PipeModeDataset that we don't want to store... So we'll create a temporary Keras .h5 file and recreate
    # the model in an empty session first:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, "model.h5")
        tf.keras.models.save_model(
            inference_model,
            tmpfilename,
            overwrite=True,
            include_optimizer=False,
        )
        clear_session_and_reseed(args.seed)
        inference_model = tf.keras.models.load_model(tmpfilename)

    # Now we can save the stripped-down TensorFlow graph ready for TFServing:
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(args.model_path, "model/1"),
        inputs={ "inputs": inference_model.input },
        outputs={ t.name: t for t in inference_model.outputs },
    )

    # Finally, we need to save our inference container code (custom I/O handlers) too.
    # Really we only need inference.py and whatever files + requirements it depends on, but for simplicity
    # we'll just copy the entire contents of this source_dir into the package:
    copy_tree(
        os.path.dirname(os.path.realpath(__file__)),
        os.path.join(args.model_path, "code"),
    )

if __name__ == "__main__":
    args = config.parse_args()

    # Set up logger:
    logging.basicConfig()
    logger = logging.getLogger()
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)

    # Start training:
    train(args)
