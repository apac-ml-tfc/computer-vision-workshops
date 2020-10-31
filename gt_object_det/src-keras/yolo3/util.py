"""Miscellaneous utility functions."""

# Python Built-Ins:
from functools import reduce

# External Dependencies:
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np
from PIL import Image

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def letterbox_image(image, size, boxes=None):
    """resize image (and boxes, if provided) with unchanged aspect ratio using padding

    Parameters
    ----------
    image : PIL.Image
    size : Tuple(width, height)
    boxes : Union[None, numpy.array, "map", "invert"]
        Provide a numpy array of boxes to map, OR "map" to return a function mapping boxes according to the
        transform, OR "invert" to return a function for the *inverse* transform, OR None

    Returns
    -------
    new_image : PIL.Image
        Padded and resized to fit in `size`
    new_boxes : Union[numpy.array, Callable[[numpy.array], numpy.array]]
        New boxes if `boxes` was an array, or a mapping function if one of the string options was used
    """
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128,128,128))
    new_image.paste(image, (dx, dy))
    
    if boxes is None:
        return new_image
    # (isinstance check stops numpy doing an expensive elementwise comparison on a matrix)
    elif isinstance(boxes, str) and boxes == "invert":
        def boxes_inverter(b):
            b_new = np.array(b)  # Copy
            b_new[:, [0, 2]] = (b_new[:, [0,2]] - dx) / scale
            b_new[:, [1, 3]] = (b_new[:, [1,3]] - dy) / scale
            return b_new
        return new_image, boxes_inverter
    else:
        def boxes_mapper(b):
            b_new = np.array(b)  # Copy
            b_new[:, [0,2]] = b_new[:, [0,2]]*scale + dx
            b_new[:, [1,3]] = b_new[:, [1,3]]*scale + dy
            return b_new
        
        if boxes is True:
            return new_image, boxes_mapper
        else:
            return new_image, boxes_mapper(boxes)

def image_to_tensor(image):
    """Convert PIL image to tensor for YOLO"""
    return np.array(image)/255.

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def preprocess_training_sample(
    image,
    boxes,
    input_shape,
    randomize=True,
    jitter=.3,
    hue=.1,
    sat=1.5,
    val=1.5,
    proc_img=True,
):
    """Ground truth pre-processing incl. optional random augmentation

    Parameters
    ----------
    image : PIL.Image
        Loaded image data (arbitrary size)
    boxes : numpy.array
        Arbitrarily many detection boxes [xmin, xmax, ymin, ymax, class_id] (absolute pixel values)
    input_shape : Tuple[int, int]
        Target network input dimensions (width, height)
    randomize : boolean, optional
        True to enable random data augmentation (see other options below)
    jitter : float, optional
        Aspect ratio jitter
    hue : float, optional
    sat : float, optional
    val : float, optional
    proc_img : boolean, optional
    """
    if not randomize:
        # Resize and prep the image and labels without any augmentation:
        resized_image, resized_boxes = letterbox_image(image, input_shape, boxes=boxes)
        return image_to_tensor(resized_image), resized_boxes

    # Else we will randomly transform to augment the data:

    iw, ih = image.size
    w, h = input_shape

    # Stretch:
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # Offset:
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new("RGB", (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # Flip (horizontal only):
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Correct the boxes:
    new_boxes = np.array(boxes)
    if len(new_boxes)>0:
        # Transform per image:
        new_boxes[:, [0,2]] = new_boxes[:, [0,2]]*nw/iw + dx
        new_boxes[:, [1,3]] = new_boxes[:, [1,3]]*nh/ih + dy
        if flip: new_boxes[:, [0,2]] = w - new_boxes[:, [2,0]]

        # Apply bounds:
        new_boxes[:, 0:2][new_boxes[:, 0:2]<0] = 0
        new_boxes[:, 2][new_boxes[:, 2]>w] = w
        new_boxes[:, 3][new_boxes[:, 3]>h] = h

        # Discard invalid (less than a pixel wide or tall):
        box_w = new_boxes[:, 2] - new_boxes[:, 0]
        box_h = new_boxes[:, 3] - new_boxes[:, 1]
        new_boxes = new_boxes[np.logical_and(box_w > 1, box_h > 1)]

    # Finally distort colours and extract image tensor data:
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(image_to_tensor(image))
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    return image_data, new_boxes
