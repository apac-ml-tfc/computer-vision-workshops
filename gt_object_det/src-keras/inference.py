"""SageMaker inference I/O functions for Keras YOLOv3

Custom functions for de/serialization as per:
https://sagemaker.readthedocs.io/en/stable/using_tf.html#create-python-scripts-for-custom-input-and-output-formats

Accept JPEGs/PNGs of arbitrary size: resize them to the model input shape, and rescale model outputs.
"""

# Python Built-Ins:
from collections import namedtuple
from io import BytesIO
import json
import logging

# External Dependencies:
import numpy as np
from PIL import Image
import requests

# Local Dependencies
from yolo3 import util

logging.basicConfig()
logger = logging.getLogger()

CustomContext = namedtuple("CustomContext", ["boxes_mapper"])

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
        return json.dumps({ "instances": [data.tolist()] }), CustomContext(None)
    elif (
        context.request_content_type == "application/x-image"
        or context.request_content_type.startswith("image/")
    ):
        logger.info("Got image request %s", context.request_content_type)
        img_raw = Image.open(BytesIO(raw_data.read()))
        logger.info("Raw image shape %s", img_raw.size)
        # TODO: Parameterize data_shape from training run shape
        img_resized, inverse_mapper = util.letterbox_image(img_raw, (416, 416), boxes="invert")
        data = np.array(util.letterbox_image(img_raw, (416, 416)))

        logger.info("Transformed image len %s, image shape %s", len(data), data.shape)
        logger.info("Parsed input shape %s", data.shape)
        return json.dumps({ "instances": [data.tolist()] }), CustomContext(inverse_mapper)
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

    if custom_context.boxes_mapper:
        # TODO: Can't rescale the bounding box outputs until we're outputting bounding boxes!
        warnmsg = (
            "Didn't adjust raw results to reflect image re-scaling, since bounding box output layer is not "
            "yet implemented!"
        )
        logger.warning(warnmsg)
        pred_parsed = json.loads(prediction.decode("utf-8"))
        pred_parsed["warning"] = warnmsg
        prediction = json.dumps(pred_parsed).encode("utf-8")

    return prediction, response_content_type
