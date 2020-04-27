"""NumPy-based utilities for post-processing YOLOv3 model outputs

TODO: Factor as much of this as possible into the Keras inference model itself!

Since the saved Keras inference model doesn't completely encapsulate bounding box post-processing, and the
TFServing container where this inference post-processing code runs only has a cut-down TF install (e.g. no
tf.keras present); we've adapted the original TF-based evaluation code into NumPy-based logic.
"""

# Python Built-Ins:
import logging

# External Dependencies:
import numpy as np
import tensorflow as tf

logging.basicConfig()
logger = logging.getLogger("postproc")


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def non_max_suppression(boxes, scores, iou_threshold):
    """Stand-in for tf.image.non_max_suppression()

    Parameters
    ----------
    boxes : Array-like, (N, 4)
        Set of [xmin, ymin, xmax, ymax] boxes... Will work same if X and Y (or min and max) inverted
    scores : Array-like, (N,)
        One confidence score per box
    iou_threshold : float
        Intersection-over-union threshold above which to suppress overlapping boxes
    
    Returns
    -------
    keep_index : np.array, (None,)
        Set of integer `boxes` indexes to keep
    """
    n_boxes = boxes.shape[0]
    assert n_boxes == scores.shape[0], (
        "Got {} boxes but {} scores: boxes and scores dimension 0 must match".format(
            n_boxes,
            scores.shape[0],
        )
    )

    # These checks enable us to handle inversion of *min and *max without IoU messing up... Could disable
    # them for performance if you're confident in your inputs:
    n_inverted_x = np.sum((boxes[:, 2] - boxes[:, 0]) < 0)
    if (n_inverted_x == n_boxes):
        logger.warning("boxes[:, 0] (xmin) all bigger than boxes[:, 2] (xmax) - inverting")
        boxes = boxes[:, [2, 1, 0, 3]]
    elif (n_inverted_x > 0):
        raise ValueError(
            "{} of {} boxes had xmin bigger than xmax: Did you supply as [x1, x2, y1, y2]?".format(
                n_inverted_x,
                n_boxes
            )
        )
    n_inverted_y = np.sum((boxes[:, 3] - boxes[:, 1]) < 0)
    if (n_inverted_y == n_boxes):
        logger.warning("boxes[:, 1] (ymin) all bigger than boxes[:, 3] (ymax) - inverting")
        boxes = boxes[:, [0, 3, 2, 1]]
    elif (n_inverted_y > 0):
        raise ValueError(
            "{} of {} boxes had ymin bigger than ymax: Did you supply as [x1, x2, y1, y2]?".format(
                n_inverted_y,
                n_boxes
            )
        )

    # Precompute box areas for efficiency:
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # Explicit axis=0 in case the scores are (N, 1) instead of just (N,); negated to sort descending:
    candidate_indexes = np.argsort(-scores, axis=0).tolist()

    # Iteratively keep candidates and exclude remaining candidates that overlap more than threshold:
    keep_indexes = []
    while len(candidate_indexes):
        index = candidate_indexes.pop()
        keep_indexes.append(index)
        if not len(candidate_indexes):
            break
        ious = get_ious(
            boxes[index],
            boxes[candidate_indexes],
            areas[index],
            areas[candidate_indexes],
        )
        # Can't use straight binary mask indexing because candidate_indexes is a list not NumPy:
        candidate_indexes = [
            c for i, c in enumerate(candidate_indexes) if ious[i] <= iou_threshold
        ]
    
    return np.array(keep_indexes, dtype=np.int32)


def get_ious(box, boxes, box_area, boxes_area):
    """Calculate Intersection-over-Union between `box` and other `boxes`

    Using pre-computed areas for efficiency.

    box and boxes MUST be in [xmin, ymin, xmax, ymax] format, or with X and Y inverted: Swapping mins and
    maxes will mess up the maths.
    """
    assert boxes.shape[0] == boxes_area.shape[0]

    # The intersection between box and boxes is the max xmin to the min xmax, and likewise for y.
    # If the specimen box's xmax is smaller than the target box's xmin, intersection saturates to 0.
    intersections = (
        np.maximum(
            0,
            # X2 - X1:
            np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])
        )
        *
        np.maximum(
            0,
            # Y2 - Y1:
            np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])
        )
    )

    unions = box_area + boxes_area - intersections
    return intersections / unions


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = np.array(feats.shape[1:3]) # height, width
    grid_y = np.tile(
        np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1],
    )
    grid_x = np.tile(
        np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1],
    )
    # Note K.concatenate uses last dimension, not first:
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = grid.astype(feats.dtype)

    logger.info("feats shape %s, num_anchors %s, num_classes %s", feats.shape, num_anchors, num_classes)
    logger.info("anchors %s", anchors)
    logger.info("grid_shape %s and real %s", grid_shape, grid.shape)
    logger.info("grid_x shape %s, grid_y shape %s", grid_x.shape, grid_y.shape)
    feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (sigmoid(feats[..., :2]) + grid) / grid_shape[::-1].astype(feats.dtype)
    box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / input_shape[::-1].astype(feats.dtype)
    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = input_shape.astype(box_yx.dtype)
    image_shape = image_shape.astype(box_yx.dtype)
    new_shape = np.round(image_shape * np.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    # Note K.concatenate uses last dimension, not first:
    boxes =  np.concatenate(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ],
        axis=-1
    )

    # Scale boxes back to original image shape.
    # Note K.concatenate uses last dimension, not first:
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, num_classes, input_shape
    )
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(
    yolo_outputs,
    anchors,
    num_classes,
    image_shape,
    max_boxes=20,
    score_threshold=.6,
    iou_threshold=.5
):
    """Evaluate YOLO model on given input and return filtered boxes.
    
    Parameters
    ----------
    yolo_outputs : List[np.array]
        The output tensor for each layer of the YOLO network (2 for tiny, 3 for standard)
    anchors : np.array(A, 2)
        Anchor boxes as used in the original model training
    num_classes : int
        Number of classes the model is configured for
    image_shape : np.array(2)
        TODO: Width, Height or Height, Width?
    max_boxes : int
        Maximum number of boxes to return **PER CLASS**
    score_threshold : float
        Minimum confidence score to retain a box (applied before non-maximum suppression)
    iou_threshold : float
        Maximum intersection-over-union above which only the highest-confidence box is retained (NMS)

    Returns
    -------
    boxes : np.array(None, 4)
        Up to num_classes * max_boxes bounding boxes, non-normalized
    scores : np.array(None)
        One confidence score per returned box
    classes : np.array(None)
        One class ID per returned box
    """
    # TODO: Can we factor num_classes out of params and infer from network output shape?
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = np.array(yolo_outputs[0].shape[1:3]) * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_outputs[l], anchors[anchor_mask[l], ...], num_classes, input_shape, image_shape
        )
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    logger.info("Got boxes shape %s", boxes.shape)
    logger.info("Got box_scores shape %s", box_scores.shape)

    mask = box_scores >= score_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        # No need to use tf.boolean_mask as numpy supports masks directly via indexing:
        class_boxes = boxes[mask[:, c], :]
        #class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = box_scores[mask[:, c], c]
        #class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        
        boxes_keep_index = non_max_suppression(class_boxes, class_box_scores, iou_threshold)[:max_boxes]
        boxes_.append(class_boxes[boxes_keep_index, :])
        scores_kept = class_box_scores[boxes_keep_index]
        scores_.append(scores_kept)
        classes_.append(np.ones_like(scores_kept, dtype=np.int32) * c)

    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
