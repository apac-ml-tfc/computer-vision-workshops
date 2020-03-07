"""SageMaker Processing script to extract frame images from videos

Some quirks of this script for demonstration purposes:

- We support both inline installation of OpenCV (for using with the SageMaker default Scikit-Learn
  container) and pre-installation (for using with a custom container where OpenCV is already
  installed, identified by setting the `OPENCV_PREINSTALLED` environment variable)
- We offer a `--frames-per-second` parameter (to demonstrate in the associated notebook how to pass
  parameters to a SageMaker processing job), but haven't actually implemented FPS support.
- We implement a naive parallelization solution (processing every Nth file) to demonstrate how a
  Processing script can identify the number of instances running the job, and which instance it's
  running in... But this should **not** be necessary for this use case: Instead use ProcessingInput
  `s3_data_distribution_type="ShardedByS3Key"` - which will split input data between nodes
  automatically, saving resources!

"""

# Python Built-Ins:
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import List
import warnings

# OpenCV requires some OS-level dependencies not in the standard container.
# For this example comparing the built-in container to a custom one, we'll use the same script file
# in both containers and set an environment variable to indicate which one we're in:
if not os.environ.get("OPENCV_PREINSTALLED"):
    subprocess.call(["apt-get", "update"])
    subprocess.call(["apt-get", "-y", "install", "libglib2.0", "libsm6", "libxext6", "libxrender-dev"])
    subprocess.call([sys.executable, "-m", "pip", "install", "opencv-python"])
    # (or `opencv-contrib-python` if contrib modules required)
else:
    print("Skipping OpenCV install due to OPENCV_PREINSTALLED env var")

# External Dependencies:
import cv2

SUPPORTED_EXTENSIONS = ("avi", "mp4")

def existing_folder_arg(raw_value:str) -> str:
    """argparse type for a folder that must already exist"""
    value = str(raw_value)
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError("%s is not a directory" % value)
    return value

def list_arg(raw_value:str) -> List[str]:
    """argparse type for a list of strings"""
    return str(raw_value).split(",")

def parse_args() -> None:
    """Load configuration from CLI and SageMaker environment variables"""
    parser = argparse.ArgumentParser(description="Extract frame images from video files")
    parser.add_argument("--input", type=existing_folder_arg,
        default="/opt/ml/processing/input/videos",
        help="Source folder of video files",
    )
    parser.add_argument("--output", type=str, default="/opt/ml/processing/frames",
        help="Target folder for saving frame images",
    )
    parser.add_argument("--frames-per-second", type=float, default=0,
        help="(Approximate) number of frames per second to save, or save every frame if 0",
    )
    parser.add_argument("--custom-sharding", action="store_true",
        help="Set to use software-based sharding, instead of assuming pre-sharded data",
    )
    
    # Unlike in training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing
    # jobs need to refer to the resource config file to determine how many instances are running
    # and which index we are:
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found: default to one instance")
        pass # Ignore

    # In case the config file is not found (e.g. for local running), the same configuration can be
    # provided through CLI for convenience:
    parser.add_argument("--hosts", type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job"
    )
    parser.add_argument("--current-host", type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job"
    )

    return parser.parse_args()

def extract_frames(src:str, dest:str, fps:float=0, shard_ix:int=0, shard_count:int=1) -> None:
    """Extract frame images from (a shard subset of) video files in src folder

    If sharding is enabled (by setting `shard_count` > 1), only every `shard_count`th file
    (alphabetically by filename) will be processed.

    Parameters
    ----------
    src : str
        Source folder containing video files
    dest : str
        Destination folder to write image files to
    fps : float
        Frames per second limit is **not implemented!** 0 = source file FPS
    shard_ix : int
        Which shard to process, for naive filename-based sharding
    shard_count : int
        Number of shards, for naive filename-based sharding
    """
    # Getting the FPS of a video is major_ver dependent in OpenCV:
    (cv_major_ver, _, _) = (cv2.__version__).split(".")
    
    if fps != 0:
        raise NotImplementedError("FPS is a parameter, but not yet implemented!")
    
    for ix, filename in enumerate(sorted(os.listdir(src))):
        # Simple/naive parallelization:
        # (Note 0 % anything = 0, so need to offset to 1-based ix)
        if (ix + 1) % shard_count != shard_ix:
            continue

        vidid, _, extension = filename.rpartition(".")
        if (extension not in SUPPORTED_EXTENSIONS):
            print(f"Skipping non-video file {filename}")
            continue
        vidid = filename.rpartition(".")[0]
        vidcap = cv2.VideoCapture(f"{src}/{filename}")
        vidfps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS if int(cv_major_ver) < 3 else cv2.CAP_PROP_FPS)
        try:
            shutil.rmtree(f"{dest}/{vidid}")
        except FileNotFoundError:
            pass
        os.makedirs(f"{dest}/{vidid}", exist_ok=True)
        count = 0
        success = True
        success, image = vidcap.read()
        if success:
            print(f"Extracting {filename}", end="")
        else:
            print(f"Error extracting {filename}: Couldn't read capture!", end="")
        while success:
            if (not count % 25):
                print(".", end="")
            cv2.imwrite(f"{dest}/{vidid}/frame-{count:08}.jpg", image)
            success, image = vidcap.read()
            count += 1
        print()
        print(f"Captured {count} frames from {filename}")
    print(f"Output to {dest}:")
    print(os.listdir(dest))

if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)
    print("Environment variables:")
    print(os.environ)

    os.makedirs(args.output, exist_ok=True)
    if args.custom_sharding:
        warnings.warn(
            "Using custom (code-based) sharding to ignore extra input files replicated between "
            "instances... This is implemented as an example of instance count monitoring only, "
            "and applications should use ProcessingInput `s3_data_distribution_type` = "
            "'ShardedByS3Key' instead to save time and resources!"
        )
        extract_frames(
            args.input,
            args.output,
            fps=args.frames_per_second,
            shard_ix=args.hosts.index(args.current_host),
            shard_count=len(args.hosts)
        )
    else:
        extract_frames(
            args.input,
            args.output,
            fps=args.frames_per_second,
        )
