"""Function defs for SageMaker MXNet container to host a GluonCV YOLOv3 object detector

This module exports functions which override the SM MXNet framework app defaults, allowing it to
read and use a GluonCV model.
"""

# Python Built-Ins:
from io import StringIO
import logging
import os

# External Dependencies:
import mxnet as mx
import numpy as np
from gluoncv.data.transforms.bbox import resize as resize_bboxes
from gluoncv.data.transforms.presets.yolo import transform_test

# Local Dependencies:
import config

logger = logging.getLogger()


def model_fn(model_dir):
    """Customized inference container function to load a Gluon model and additional config file.

    Exporting model_fn overrides the default SageMaker MXNet container model loading behaviour
    when a worker is first started up.

    This implementation is designed to load exactly one Hybrid Gluon model saved with `.export()`:
    i.e. a *-symbol.json structure file and corresponding *-####.params weights file.

    We don't mind what the model name (*) is, so long as there's only one present, and we take the
    latest epoch (####) if there are multiple params files.

    We assume it's a single-input model.
    """
    files = os.listdir(model_dir)
    symbol_files = list(filter(lambda f: f.endswith("-symbol.json"), files))
    n_models = len(symbol_files)
    logger.info(f"Found {n_models} model symbol files: {symbol_files}")

    if n_models < 1:
        raise ValueError("No *-symbol.json Gluon export files found in model folder")
    elif n_models > 1:
        raise NotImplementedError(
            f"Found {n_models} models: Multi-model orchestration not implemented at this level. "
            "Try using SageMaker Multi-Model Endpoints instead"
        )

    symbol_file = symbol_files[0]
    model_prefix = symbol_file.rpartition("-")[0]

    # Parameter files look like prefix-####.params: with a 4-zero-padded epoch number.
    # We'll use the most recent provided epoch.
    param_files = list(filter(
        lambda f: f.startswith(model_prefix) and f.endswith(".params"),
        files
    ))
    logger.info(f"Found {len(param_files)} model parameter files - choosing latest: {param_files}")

    # The epoch number could technically be >4 digits if more than 9999 epochs were trained, so
    # let's be more rigorous than just sorting alphabetically... Just in case:
    param_epochs = list(map(
        lambda f: int(f.rpartition(".")[0].rpartition("-")[2]),
        param_files
    ))
    ix_latest_params = np.argmax(param_epochs)
    param_file = param_files[ix_latest_params]
    logger.info(f"Loading {symbol_file} | {param_file}")

    net = mx.gluon.SymbolBlock.imports(
        os.path.join(model_dir, symbol_file),
        # Single-input model will have its one input named "data", per HybridBlock.export:
        ["data"],
        os.path.join(model_dir, param_file),
    )

    # Finally, load the associated inference configuration file:
    inference_config = config.InferenceConfig.load(
        os.path.join(model_dir, f"{model_prefix}-inference-config.json")
    )
    return (net, inference_config)


def transform_fn(model, raw_data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param model: Tuple of the Gluon model and inference config objects
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # Unpack the neural net and the additional algo config:
    net, inference_config = model
    # First step: Read the input data:
    resized = False
    batch = False  # TODO: Support batch properly for numpy inputs
    if input_content_type == "application/x-npy":
        logger.info(f"Got raw tensor request {input_content_type}")
        stream = BytesIO(raw_data)
        data = mx.nd.array(np.load(stream))
        logger.info(f"Parsed tensor shape {data.shape}")
        n_data_dims = len(data.shape)
        if n_data_dims == 4:
            batch = True
        elif n_data_dims != 3:
            raise ValueError(
                "Expect a [ndata x nchannels x width x height] (YOLO-normalized) batch image array, "
                f"or a single image with batch dimension omitted... Got shape {data.shape}"
            )
    elif input_content_type == "application/x-image" or input_content_type.startswith("image/"):
        logger.info(f"Got image request {input_content_type}")
        img_raw = mx.image.imdecode(bytearray(raw_data))
        img_raw_shape = img_raw.shape[:-1] # Skip the channels dimension
        logger.info(f"Raw image shape {img_raw.shape}")

        data, _ = transform_test(img_raw, short=inference_config.image_size)
        img_transformed_shape = data.shape[2:] # Channels dimension is leading now
        resized = True

        logger.info(f"Transformed image len {len(data)}, image shape {data.shape}")
        logger.info(f"Parsed input shape {data.shape}")
    else:
        logger.error(f"Got unexpected request content type {input_content_type}")
        raise ValueError(f"Unsupported request content type {input_content_type}")

    # Run the data through the network:
    # YOLOv3 expects [ndata x nchannels x width x height], normalized pixel values and outputs:
    # Class IDs [ndata x boxlimit x 1], floating point dtype but integer values
    # Confidence scores [ndata x boxlimit x 1], float 0-1
    # Bounding boxes [ndata x boxlimit x 4], float absolute pixel (xmin, ymin, xmax, ymax)
    #
    # Limit of 100 boxes by default, padded with -1s for null detections / no boxes.
    ids, scores, bboxes = net(data)
    logger.info(
        f"Model output: ids {ids.shape}[{ids.dtype}], scores {scores.shape}[{scores.dtype}], "
        f"bboxes {bboxes.shape}[{bboxes.dtype}]"
    )

    # Resize and normalize the output bounding boxes
    if resized:
        bboxes = resize_bboxes(
            bboxes,
            in_size=img_transformed_shape,
            out_size=img_raw_shape
        )
        bboxes = resize_bboxes(
            bboxes,
            in_size=img_raw_shape,
            out_size=(1., 1.)
        )
        logger.info("Normalized bounding boxes")

        # Annoyingly, gluoncv.data.transforms.bbox.resize() messes about with the bboxes dtype
        # by casting .astype(float) - converting from original 32 bit to 64. concat()ing the
        # results requires shared dtype, so we'll do the same wasteful transform on the others:
        ids = ids.astype(float)
        scores = scores.astype(float)

    # Convert to numpy after stacking, because mx.nd doesn't implement tolist or savetxt for our
    # output serialization:
    stacked_np = mx.nd.concat(ids, scores, bboxes, dim=2).asnumpy()
    # [ndata x boxlimit x (1+1+4)]

    # Rather than a padded matrix, we'll return only what boxes we've found.
    # Note, this means that you won't be able to just interpret batch results as a 3D matrix!
    box_exists = stacked_np[:,:,0] >= 0 # Boxes exist where class ID is not -1

    if output_content_type == "text/csv":
        if batch:
            raise NotImplementedError(
                "Haven't implemented CSV output for batch requests yet! Use SingleRecord splitting"
            )
        else:
            s = StringIO()
            np.savetxt(s, stacked_np[0, box_exists[0]], delimiter=",")
            return (
                s.getvalue(),
                output_content_type
            )
    else:
        if output_content_type not in ("application/json", None):
            logger.error(
                f"Got unexpected output content type {output_content_type}: Returning JSON"
            )
        return (
            {
                "prediction": (
                    [
                        stacked_np[ix_img, box_exists[ix_img]].tolist()
                        for ix_img in range(stacked_np.shape[0])
                    ] if batch
                    else stacked_np[0, box_exists[0]].tolist()
                )
            },
            "application/json"
        )
