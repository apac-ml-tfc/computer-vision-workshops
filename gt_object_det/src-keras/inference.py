"""SageMaker inference I/O functions for Keras YOLOv3

Custom functions for de/serialization as per:
https://sagemaker.readthedocs.io/en/stable/using_tf.html#create-python-scripts-for-custom-input-and-output-formats

Accept JPEGs/PNGs of arbitrary size: resize them to the model input shape, and rescale model outputs.
"""

# Python Built-Ins:
from collections import defaultdict, namedtuple
from io import BytesIO
import json
import logging

# External Dependencies:
import numpy as np
from PIL import Image
import requests

# Local Dependencies
from yolo3 import postproc, util

logging.basicConfig()
logger = logging.getLogger()

# TODO: Double-check all our size and box orders tie up (width-major or height-major?)
# raw_image_shape is a width,height tuple (same as PIL Image.size produces)
CustomContext = namedtuple("CustomContext", ["boxes_mapper", "raw_image_shape"])

def handler(data, context):
    """Overall request handler

    Scaling images for the TF model means we need to re-scale the bounding box outputs too, so we have to
    implement an end-to-end handler() rather than independent input_handler() and output_handler() - in order
    to keep track of the original image size.

    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input, custom_context = _input_handler(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _output_handler(response, context, custom_context)


def _input_handler(raw_data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API

    Implements the standard SM TF input_handler() interface, apart from returning an extra custom_context
    object to store the scaling factors.

    Args:
        raw_data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        processed_input (dict): a JSON-serializable dict that contains request body and headers
        custom_context (CustomContext): To be passed to the output handler
    """
    # First step: Read the input data:
    batch = False  # TODO: Support batch properly for numpy inputs
    if context.request_content_type == "application/x-npy":
        logger.info("Got raw tensor request %s", context.request_content_type)
        stream = BytesIO(raw_data.read())
        data = np.load(stream)
        logger.info("Parsed tensor shape %s", data.shape)
        n_data_dims = len(data.shape)
        if n_data_dims == 4:
            batch = True
        elif n_data_dims != 3:
            raise ValueError(
                "Expect a [ndata x nchannels x width x height] (YOLO-normalized) batch image "
                "array, or a single image with batch dimension omitted... Got shape {}".format(data.shape)
            )
        return (
            json.dumps({ "instances": [data.tolist()] }),
            CustomContext(None, (416, 416)),
        )
    elif (
        context.request_content_type == "application/x-image"
        or context.request_content_type.startswith("image/")
    ):
        logger.info("Got image request %s", context.request_content_type)
        img_raw = Image.open(BytesIO(raw_data.read()))
        logger.info("Raw image shape %s", img_raw.size)
        # TODO: Parameterize data_shape from training run shape
        if img_raw.size[0] == 416 and img_raw.size[1] == 416:
            img_resized = img_raw
            inverse_mapper = None
        else:
            img_resized, inverse_mapper = util.letterbox_image(img_raw, (416, 416), boxes="invert")

        data = np.array(img_resized)
        logger.info("Transformed image shape %s", data.shape)
        return json.dumps({ "instances": [data.tolist()] }), CustomContext(inverse_mapper, img_raw.size)
    else:
        logger.error("Got unexpected request content type %s", input_content_type)
        raise ValueError("Unsupported request content type {}".format(input_content_type))


def _output_handler(data, context, custom_context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))

    response_content_type = context.accept_header
    prediction = data.content

    # The TFServing result.predictions is a list (batch dim) of dicts (output name) of array lists.
    # We unpack the batch dimension into each output's array, and return an alphabetical list of the
    # outputs (which are named according to the output layer index):
    preds_parsed = json.loads(prediction.decode("utf-8"))["predictions"]
    output_dict = defaultdict(list)
    for prediction in preds_parsed:
        for output_id in prediction:
            output_dict[output_id] = output_dict[output_id] + [prediction[output_id]]
    
    output_ids = sorted(output_dict.keys())
    logger.info("Output IDs %s", output_ids)
    raw_output = [np.stack(output_dict[k], axis=0) for k in output_ids]

    logger.info(
        "Raw output is [{}]".format((
            ", ".join([
                "Array<{}, {}>".format(o.shape, o.dtype)
                for o in raw_output
            ])
        ))
    )

    # TODO: postproc.yolo_eval expects batch dim but is not actually set up to calculate on batches
    boxes, scores, classes = postproc.yolo_eval(
        raw_output,
        np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]),
        2,
        np.array([416, 416]),
        max_boxes=30,
        score_threshold=.2,
        iou_threshold=.5,
    )

    # Re-scale the bounding boxes if the image was resized in pre-processing:
    if custom_context.boxes_mapper:
        boxes = custom_context.boxes_mapper(boxes)

    # Finally, normalize the coords and combine into a single matrix for consistency with other algos:
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / custom_context.raw_image_shape[0]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / custom_context.raw_image_shape[1]
    result = {
        "predictions": np.concatenate(
            [
                classes.reshape([-1, 1]),
                scores.reshape([-1, 1]),
                boxes,
            ],
            axis=1
        ).tolist()
    }

    return json.dumps(result).encode("utf-8"), response_content_type
