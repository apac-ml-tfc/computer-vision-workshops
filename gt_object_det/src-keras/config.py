"""SageMaker training job configuration loading
"""

# Python Built-Ins:
import argparse
import json
import logging
import os


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
    parser = argparse.ArgumentParser(description="Train YOLO object detector.")

    ## Network parameters:
    parser.add_argument("--data-shape", type=int, default=hps.get("data-shape", 416),
        help="Input data shape, use a multiple of 32 e.g. 320, 416, 608... "
    )
    parser.add_argument("--num-classes", type=int, default=hps.get("num-classes", 1),
        help="Number of classes in training data set."
    )

    ## Training process parameters:
    parser.add_argument("--batch-size", type=int, default=hps.get("batch-size", 4),
        help="Training mini-batch size"
    )
    parser.add_argument("--epochs", type=int, default=hps.get("epochs", 1),
        help="The maximum number of passes over the training data."
    )
    parser.add_argument("--epochs-stabilize", type=int, default=hps.get("epochs-stabilize", 1),
        help="The number of epochs to pre-train with frozen layers to stabilize (must be <= epochs)."
    )
    parser.add_argument("--lr", "--learning-rate", type=float,
        default=hps.get("lr", hps.get("learning-rate", 0.0001)),
        help="Learning rate (main training cycle)"
    )
    parser.add_argument("--lr-pretrain", type=float, default=hps.get("lr-pretrain"),
        help="Learning rate for initial stabilization phase: 10x standard learning rate if not set"
    )
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)."
    )

    ## I/O Settings:
    parser.add_argument("--checkpoint-dir", type=str,
        default=hps.get("checkpoint-dir", "/opt/ml/checkpoints")
    )
    parser.add_argument("--checkpoint-interval", type=int,
        default=hps.get("checkpoint-interval", 0),
        help="Epochs between saving checkpoints (set 0 to disable)"
    )
    parser.add_argument("--output-data-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    # IGNORE this guff from the TensorFlow container:
    # https://github.com/aws/sagemaker-tensorflow-container/issues/340
    parser.add_argument("--model_dir", type=str, default="")
    # USE THIS INSTEAD for saving models:
    parser.add_argument("--model-path", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--num-samples-train", type=int, default=hps.get("num-samples-train"),
        help="Number of samples in training set (required for Pipe Mode generator compatibility).",
    )
    parser.add_argument("--num-samples-validation", type=int, default=hps.get("num-samples-validation"),
        help="Number of samples in validation set (required for Pipe Mode generator compatibility).",
    )
    parser.add_argument("--darknet", type=str, default=os.environ.get("SM_CHANNEL_DARKNET"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )

    args = parser.parse_args()

    ## Post-argparse validations/transformations:
    assert args.epochs > 0, "How am I supposed to train for 0 epochs??"
    assert args.epochs_stabilize <= args.epochs, (
        f"Stabilization epochs ({args.epochs_stabilize}) must be <= total epochs ({args.epochs})"
    )
    if args.lr_pretrain is None:
        args.lr_pretrain = args.lr * 10;
    if (args.darknet):
        cfg_files = list(filter(lambda s: s.endswith(".cfg"), os.listdir(args.darknet)))
        n_cfg_files = len(cfg_files)
        assert n_cfg_files == 1, (
            "darknet channel, if provided, must contain exactly 1 .cfg/.weights file pair: A previously"
            f"trained darknet model to load into Keras. Got {n_cfg_files} .cfg files."
        )
        file_root = cfg_files[0][:-4] # Remove .cfg extension
        assert os.path.isfile(f"{args.darknet}/{file_root}.weights"), (
            "darknet channel, if provided, must contain exactly 1 .cfg/.weights file pair: A previously"
            f"trained darknet model to load into Keras. Found {cfg_files[0]} but not {file_root}.weights"
        )
        # Set the argument to the file root:
        args.darknet = f"{args.darknet}/{file_root}"

    return args
