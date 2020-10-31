"""Configuration utilities for SageMaker Gluon YOLOv3"""

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
    """Training job command line / environment variable configuration parser"""
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
    parser.add_argument("--num-samples-train", type=int, default=hps.get("num-samples-train"),
        help="Number of samples in training set (required for GluonCV compatibility).",
    )
    parser.add_argument("--num-samples-validation", type=int,
        default=hps.get("num-samples-validation"),
        help="Number of samples in validation set (required for GluonCV compatibility).",
    )
    parser.add_argument("--save-interval", type=int, default=hps.get("save-interval", 10),
        help="Saving parameters epoch interval, best model will always be saved."
    )

    args = parser.parse_args()

    # Post-argparse validations:
    if not args.num_samples_train:
        raise argparse.ArgumentError(
            "num-samples-train must be provided via CLI or environment variable, for "
            "compatibility with GluonCV (which requires Datasets to have known length)"
        )
    if args.validation and not args.num_samples_train:
        raise argparse.ArgumentError(
            "num-samples-validation must be provided via CLI or environment variable, for "
            "compatibility with GluonCV (which requires Datasets to have known length)"
        )
    args.resume = args.resume.strip()
    return args


class InferenceConfig:
    def __init__(self, image_size: int=0):
        """Container for inference-time algorithm configuration (beyond the DNN itself)

        This class will be `save()`d into the model.tar.gz as a JSON file and `load()`ed at inference time

        Parameters
        ----------
        image_size : int
            Pixel width (=height) of the neural network input
        """
        self.image_size = image_size

    def __str__(self):
        """Compact string representation for printing to console"""
        return "InferenceConfig({})".format(
            ", ".join([
                "{}={}".format(k, v)
                for (k, v) in self.__dict__.items()
            ])
        )

    def __repr__(self):
        """JSON serialization for loading/saving to files"""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            content = json.loads(f.read())
            return cls(**content)

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(repr(self))
