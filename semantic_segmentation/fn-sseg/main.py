"""Lambda function to proxy SageMaker Semantic Segmentation endpoint into nice human-viewable images"""

# Python Built-Ins:
import base64
import io
import json
import logging
import os

# External Dependencies:
import boto3
from expiringdict import ExpiringDict
from matplotlib.pyplot import get_cmap
import numpy as np
from PIL import Image
import sagemaker

log_level = os.environ.get("LOG_LEVEL", logging.INFO)
logger = logging.getLogger()
logger.setLevel(log_level)

# Local Dependencies:
from deserializers import ImageDeserializer, HackyProtobufDeserializer

# Environment Configuration:
default_endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT")
default_target_class_id = int(os.environ.get("TARGET_CLASS_ID", 0))
default_mask_color = [float(x) for x in os.environ.get("MASK_COLOR", "0,1,1").split(",")]
default_mask_colormap = os.environ.get("MASK_COLORMAP")
default_probabilistic = (
    True if os.environ.get("PREDICT_PROBABILITIES", "no").lower() in ("1", "t" "true", "yes", "y") else False
)

s3 = boto3.client("s3")

# We'll cache our Predictors for performance, but expire them every now and then so refreshes are forced too:
predictor_cache = ExpiringDict(max_len=1, max_age_seconds=180)

def get_predictor(endpoint_name: str, probabilistic: bool=False):
    cache_key = f"{endpoint_name}/{'png' if probabilistic else 'recordio'}"
    predictor = predictor_cache.get(cache_key)
    if predictor is None:
        predictor = sagemaker.predictor.Predictor(
            endpoint_name,
            serializer=sagemaker.serializers.IdentitySerializer('image/jpeg'),
            deserializer=HackyProtobufDeserializer() if probabilistic else ImageDeserializer(),
        )
        predictor_cache[cache_key] = predictor
    return predictor


def generate_mask(model_result: np.array, alpha, mask_color, mask_colormap, probabilistic, target_class_id):
    """Human-viewable mask generation (see handler doc for details)"""
    if mask_colormap:
        cmap_name, _, cmap_config = mask_colormap.partition("|")
        cmap = get_cmap(cmap_name)
        if probabilistic:
            mask = cmap(model_result[target_class_id, :, :])
            # Apply alpha:
            if cmap_config.lower().startswith("dynamic"):
                mask[:, :, 3] = model_result[target_class_id, :, :] * alpha
            else:
                mask[:, :, 3] = alpha
        else:
            cmap_min, _, cmap_max = cmap_config.partition("|")
            is_dynamic = "dynamic" in cmap_min
            if cmap_min == "dynamic":
                cmap_min, _, cmap_max = cmap_max.partition("|")
            if cmap_min:
                cmap_min = int(cmap_min)
            else:
                cmap_min = np.min(model_result)
            if cmap_max:
                cmap_max = int(cmap_max)
            else:
                cmap_max = np.max(model_result)
            mask = cmap((model_result - cmap_min) / (cmap_max - cmap_min))
            # Apply alpha:
            if is_dynamic:
                mask[:, :, 3] = (model_result == target_class_id) * alpha
            else:
                mask[:, :, 3] = alpha
    else:
        mask_color_4channel = np.append(mask_color, [1.])
        imshape = model_result.shape[1:] if probabilistic else model_result.shape
        mask = np.tile(
            mask_color_4channel[np.newaxis, np.newaxis, :],
            list(imshape) + [1],
        )
        # Apply alpha:
        if probabilistic:
            mask[:, :, 3] = model_result[target_class_id, :, :] * alpha
        else:
            mask[:, :, 3] = (model_result == target_class_id) * alpha
    return mask


def handler(event, context):
    """SageMaker Semantic Segmentation endpoint proxy for nice human-viewable image outputs

    Call a deployed SageMaker Semantic Segmentation model and return a human-viewable PNG image for a
    specific target class ID (instead of the raw class indexed PNG or recordIO confidence tensor).

    Parameters
    ----------
    event :
        AWS Lambda event object with properties:
            - "EndpointName" : str, optional
                Deployed SageMaker endpoint name. SAGEMAKER_ENDPOINT env var will be used if not supplied
            - "Probabilistic": bool, optional
                If True, the full confidence scores will be requested from the model and the returned map
                will show the confidence of TargetClassId at each pixel. If False, the returned map will only
                indicate which pixels TargetClassId was the highest-confidence prediction (which typically 
                delivers faster model inference and lower Lambda memory usage).
                Default is False unless env var PREDICT_PROBABILITIES is '1'/'t'/'true'/'yes'/'y'.
            - "TargetClassId": int, optional
                Numeric ID of the class to highlight. Defaults to TARGET_CLASS_ID env var or 0
            - "Alpha": float, optional
                Multiplier for the alpha (opacity) channel of the returned mask image. If 1.0 (default),
                non-Probabilistic predictions return fully opaque pixels in regions where the TargetClassId
                is strongest-predicted. 0.0 will result in a fully-transparent image (probably not wanted!).
            - "MaskColor": List[float, float, float], optional
                0-1.0 RGB color to indicate for regions where the TargetClassId is present. Defaults to
                MASK_COLOR env var (set as comma-separated list e.g. 0.0,1.0,1.0) or cyan.
            - "MaskColorMap": str, optional
                Name of a matplotlib colormap to apply to the prediction result, overriding MaskColor. On
                Probabilistic predictions, the confidence for TargetClassId will be colormapped, and
                pixel-wise transparency will be set by confidence if '|dynamic' is appended to the supplied
                MaskColorMap name (otherwise transparency will be uniform as set by Alpha). On
                non-Probabilistic predictions, TargetClassId will be ignored and class IDs will be mapped.
                Append |dynamic to highlight *only* TargetClassId regions (rest transparent), and then
                integers e.g. |0|24 to explicitly set the min and max extents of the map, or else the IDs
                found in the particular image will be used to set the extents: e.g. 'inferno|dynamic|0|24'
            - "ImageBase64": str, optional
                Base64-encoded JPEG input image data, if supplying the image directly in the request payload.
                (Remember the Lambda payload size limit!).
            - "ImageS3Uri": str, optional
                URI string like s3://DOC-EXAMPLE-BUCKET/my/image.jpg, if inferencing on an image in S3.
    context :
        AWS Lambda context

    Returns
    -------
    result : dict
        JSON-serializable dict with properties:
            - "MaskBase64": str
                Base64-encoded PNG (typically translucent) of the human-viewable mask image
    """
    endpoint_name = event.get("EndpointName", default_endpoint_name)
    if endpoint_name is None:
        raise ValueError("EndpointName not provided in event and SAGEMAKER_ENDPOINT env var not set")
    probabilistic = event.get("Probabilistic", default_probabilistic)
    target_class_id = event.get("TargetClassId", default_target_class_id)
    alpha = event.get("Alpha", 1.0)
    mask_color = np.array(event.get("MaskColor", default_mask_color))
    mask_colormap = event.get("MaskColorMap", default_mask_colormap)
    if not mask_colormap and (len(mask_color) != 3):
        raise ValueError("MaskColor (or MASK_COLOR env var) must be a list of 3 RGB values between 0-1")

    logger.info({
        "EndpointName": endpoint_name,
        "Probabilistic": probabilistic,
        "TargetClassId": target_class_id,
        "Alpha": alpha,
        "MaskColor": mask_color,
        "MaskColorMap": mask_colormap
    })

    if "ImageBase64" in event:
        img = base64.b64decode(event["ImageBase64"])
        del event["ImageBase64"]
    elif "ImageS3Uri" in event:
        input_uri = event["ImageS3Uri"]
        if not input_uri.lower().startswith("s3://"):
            raise ValueError("ImageS3Uri must be a valid S3 URI e.g. s3://DOC-EXAMPLE-BUCKET/my/image.jpg")
        imgbucket, _, imgkey = input_uri[5:].partition("/")
        logger.info(f"Fetching s3://{imgbucket}/{imgkey}")
        imgs3response = s3.get_object(Bucket=imgbucket, Key=imgkey)
        img = imgs3response["Body"].read()
    else:
        raise ValueError("Must supply either ImageBase64 or ImageS3Uri in request!")

    # Get the prediction from SageMaker endpoint:
    predictor = get_predictor(endpoint_name, probabilistic)
    model_result = predictor.predict(img)

    # Generate mask:
    print("Generating mask")
    mask = generate_mask(model_result, alpha, mask_color, mask_colormap, probabilistic, target_class_id)

    print("Loading result image")
    mask_img = Image.fromarray((255 * mask).astype(np.uint8))
    result_buffer = io.BytesIO()
    print("Converting to PNG")
    mask_img.save(result_buffer, format="PNG")
    result_buffer.seek(0)
    print("Result Ready")
    # TODO: Option to save out to S3 and return URI
    return {
        "MaskBase64": base64.b64encode(result_buffer.read()).decode('utf-8')
    }
