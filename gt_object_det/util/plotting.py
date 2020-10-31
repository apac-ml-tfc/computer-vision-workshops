# Built-Ins:
import random
import warnings

# External Dependencies:
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def visualize_detection(img_file, dets, classes=[], thresh=0.6, normalized_coords=True):
    """
    visualize detections in one image
    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
        each row is one object
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """

    # Check we can open the image file up front:
    img = mpimg.imread(img_file)
    # Cast to numpy in case we received a list, and make a copy to allow in-place edits:
    dets = np.array(dets)
    n_dets = len(dets)
    assert n_dets == 0 or (len(dets.shape) == 2 and dets.shape[1] == 6), (
        "dets must be an Nx6 matrix of [id, score, x1, y1, x2, y2] rows. Got shape {}".format(dets.shape)
    )

    fig = plt.figure()
    plt.imshow(img)

    height = img.shape[0]
    width = img.shape[1]

    if n_dets > 0:
        if normalized_coords:
            dets[..., 2:6] = dets[..., 2:6] * np.array([[width, height, width, height]])
        boxes_saturated = np.maximum(
            np.array([[-.1*width, -.1*height, -.1*width, -.1*height]]),
            np.minimum(
                np.array([[1.1*width, 1.1*height, 1.1*width, 1.1*height]]),
                dets[..., 2:6]
            )
        )
        if np.any(boxes_saturated != dets[..., 2:6]):
            # Otherwise we could try to create a massive pixel buffer in memory and crash the kernel:
            warnings.warn(
                "Large boxes will be saturated to +/- 10% outside image coord range to render safely"
            )
            dets[..., 2:6] = boxes_saturated
            # Set all limits to maintain aspect ratio
            plt.gca().set_xlim((-.1*width, 1.1*width))
            plt.gca().set_ylim((1.1*height, -.1*height))

    colors = dict()
    ax = plt.gca()
    for det in dets:
        (klass, score, x0, y0, x1, y1) = det
        if score < thresh:
            continue
        cls_id = int(klass)
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colors[cls_id], linewidth=3.5)
        ax.add_patch(rect)
        class_name = str(cls_id)
        if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
        ax.text(
            x0,
            y0 - 2,
            "{:s} {:.3f}".format(class_name, score),
            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
            color="white",
            fontsize=12,
        )
    plt.show()
