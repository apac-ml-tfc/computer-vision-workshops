"""Data loading/pre-processing utilities
"""
# Python Built-Ins:
from io import BytesIO
import json

# External Dependencies:
import numpy as np
from PIL import Image
import tensorflow as tf

# Local Dependencies:
from yolo3.model import preprocess_true_boxes
from yolo3.util import preprocess_training_sample


def get_tf_parse_mapper(
    input_shape,
    randomize=True,
    max_boxes=20,
    jitter=.3,
    hue=.1,
    sat=1.5,
    val=1.5,
    proc_img=True,
):
    """Create a TF Dataset Map fn from batched AugmentedManifest records to a parsed detection example
    """
    def tf_map_detection_record(fields):
        def tf_parse_detection_record(fields):
            #fieldsjson = tf.map_fn(lambda field: field[0] == b"{", fields)
            fieldsjson = [field[0] == b"{"[0] for field in fields]  # Binary in Python is weird...
            njsonfields = sum(fieldsjson)
            if njsonfields != 1:
                raise ValueError(
                    f"Record had {njsonfields} JSON annotation fields out of {len(fields)} total: "
                    "Expected exactly one"
                )
            # Take first JSON and first non-JSON field to be the header and the image, respectively:
            label = json.loads(fields[fieldsjson.index(True)])
            raw_img = Image.open(BytesIO(fields[fieldsjson.index(False)]))
            #raw_img = tf.image.decode_jpeg(fields[0])

            raw_boxes = np.array(
                [[
                    ann["left"],
                    ann["top"],
                    (ann["left"] + ann["width"]),
                    (ann["top"] + ann["height"]),
                    ann["class_id"],
                ] for ann in label["annotations"]],
                dtype="float64",
            )

            # Scale (and optionally randomize/augment) the image and boxes to target input_shape:
            img, boxes = preprocess_training_sample(
                raw_img, raw_boxes, input_shape, randomize=randomize, jitter=jitter, hue=hue, sat=sat,
                val=val, proc_img=proc_img
            )
            return (img, boxes)

        img, boxes = tf.py_func(tf_parse_detection_record, [fields], [tf.float64, tf.float64])
        img.set_shape([input_shape[0], input_shape[1], 3])
        boxes.set_shape([None, 5])

        # Truncate and pad to max_boxes:
        boxes_padded = boxes[:max_boxes, ...]
        paddings = [
            [0, max_boxes - tf.shape(boxes_padded)[0]],  # Append zeros on dimension 0
            [0, 0]  # No padding on dimension 1
        ]
        boxes_padded = tf.pad(boxes_padded, paddings, "CONSTANT", constant_values=0)
        return (img, boxes_padded)

    return tf_map_detection_record


def get_tf_train_batch_mapper(batch_size, input_shape, anchors, num_classes):
    """Construct a TF Dataset.map(function) to pre-process batches of images+labels

    The returned mapper function converts from basic parsed image/boxes pairs to training model inputs
    """
    # TODO: Find a way to make TensorFlow accept this being dynamic, to support TinyYOLO:
    num_layers = len(anchors)//3
    assert num_layers == 3, "Can't get dynamic shape, what a pity (PITA)"
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]  # if num_layers==3 else [[3,4,5], [1,2,3]]
    grid_shapes = [np.array(input_shape)//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    output_shapes = [
        (batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes)
        for l in range(num_layers)
    ]
    
    def tf_map_train_batch_transform(imgs, boxes):
        """TF Dataset.map() function to pre-process batches of images+labels

        Essentially just a tf.py_func wrapper around a Python function
        """
        def py_map_train_batch_transform(imgs, boxes):
            y_true = preprocess_true_boxes(boxes, input_shape, anchors, num_classes)
            items = (imgs, *y_true)
#             # See for yourself that the return shapes match the advertised below:
#             print("NP batch shapes:")
#             for item in items:
#                 print(item.shape)
            return items
        
        imgs, y1, y2, y3 = tf.py_func(
            py_map_train_batch_transform,
            [imgs, boxes],
            [tf.float64, tf.float32, tf.float32, tf.float32]
        )
        imgs.set_shape((batch_size,) + input_shape + (3,))
        y1.set_shape(output_shapes[0])
        y2.set_shape(output_shapes[1])
        y3.set_shape(output_shapes[2])
#         print("TF batch shapes:")
#         print(imgs.shape)
#         print(y1.shape)
#         print(y2.shape)
#         print(y3.shape)
        return ((imgs, y1, y2, y3), np.zeros(batch_size))
    return tf_map_train_batch_transform
