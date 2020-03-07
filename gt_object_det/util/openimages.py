"""Demo utilities for Open Images dataset

The Open Images Dataset V4 is created by Google Inc. The annotations are licensed by Google Inc.
under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license. The following paper
describes Open Images V4 in depth; from the data collection and annotation to detailed statistics
about the data and evaluation of models trained on it.

A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov,
M. Malloci, T. Duerig, and V. Ferrari. The Open Images Dataset V4: Unified image classification,
object detection, and visual relationship detection at scale. arXiv:1811.00982, 2018.

These utilities simplify searching (the smaller, test portion of) the dataset for images containing
objects by name, since Open Images uses hierarchical, alphanumeric ID codes instead of
human-readable class names.
"""

# Python Built-Ins:
from collections import defaultdict
import csv
import json
import logging
import os
from typing import Dict, Iterable, Mapping, Set, Tuple
import urllib
import warnings

logger = logging.getLogger("util.openimages")
logger.setLevel(logging.INFO) # Feel free to change to logging.DEBUG to see more detailed output


def download_openimages_metadata(target_folder: str) -> Tuple[str, str, str]:
    """Download metadata files for the Open Images (v4 test) dataset
    
    Creates a class hierarchy JSON, class descriptions CSV, and image annotations CSV (bounding
    boxes) file in the `target_folder`, and returns the locations of the files.
    
    Returns
    -------
    annotations : str
        Bounding boxes CSV path
    class_descriptions : str
        Class name - ID mapping CSV path
    ontology : str
        Class hierarchy JSON file path
    """
    os.makedirs(target_folder, exist_ok=True)
    annotations = f"{target_folder}/annotations-bbox.csv"
    class_descriptions = f"{target_folder}/class-descriptions.csv"
    ontology = f"{target_folder}/ontology.json"

    urllib.request.urlretrieve(
        "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv",
        filename=annotations,
    )
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/openimages/2018_04/class-descriptions.csv",
        filename=class_descriptions,
    )
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json",
        filename=ontology,
    )

    return annotations, class_descriptions, ontology


def get_all_subclasses(class_id: str, tree: Dict) -> Set[str]:
    """Get the set of `class_id` and all matching subclasses from hierarchy `tree`
    
    Parameters
    ----------
    class_id : str
        Open Images class ID string (e.g. "/m/01yrx")
    tree : Dict
        Loaded Open Images hierarchy JSON object
    
    Returns
    -------
    classes : Set[str]
        Set of class IDs including class_id and all child classes
    """
    def all_subtree_class_ids(subtree: Dict) -> Set[str]:
        if ("Subcategory" in subtree):
            return set([subtree["LabelName"]]).union(
                *[all_subtree_class_ids(s) for s in subtree["Subcategory"]]
            )
        else:
            return set([subtree["LabelName"]])
    if (tree["LabelName"] == class_id):
        return all_subtree_class_ids(tree)
    elif "Subcategory" in tree:
        return set().union(*[get_all_subclasses(class_id, s) for s in tree["Subcategory"]])
    else:
        return set()


def class_names_to_openimages_ids(
    class_names: Iterable[str],
    descriptions_file: str,
    ontology_file: str,
) -> Mapping[str, Set[str]]:
    """(Case insensitive) lookup of Open Images IDs for label names
    
    For each class_name, find the equivalent Open Images class ID and all sub-class IDs (since
    Open Images is a hierarchical ontology)
    
    Parameters
    ----------
    class_names : Iterable[str]
        Label names e.g. "bird" to look for
    descriptions_file : str
        Location of Open Images "class descriptions" CSV
    ontology_file : str
        Location of Open Images label "hierarchy" JSON
    
    Returns
    -------
    results : Dict[str, Set[str]]
        Mapping from each class name to the set of Open Images class IDs representing it
    """
    # The class list is really long, so let's stream it instead of loading as dataframe:
    class_root_ids = { s: None for s in class_names }
    classes_lower_notfound = { s.lower(): s for s in class_names }
    with open(descriptions_file, "r") as f:
        for row in csv.reader(f):
            row_class_lower = row[1].lower()
            match = classes_lower_notfound.get(row_class_lower)
            if (match is not None):
                class_root_ids[match] = row[0]
                del classes_lower_notfound[row_class_lower]
                if (len(classes_lower_notfound) == 0):
                    logger.debug("Class name -> root ID mapping done")
                    break

    logger.debug("Found class root IDs:")
    logger.debug(class_root_ids)
    if len(classes_lower_notfound):
        raise ValueError(
            f"OpenImages IDs not found for these class names: "
            f"{[v for (k,v) in classes_lower_notfound.items()]}"
        )

    # Next, we recurse down the ontology from these root classes to capture any child classes.
    # (Note that actually "boot" and "cat" are leaf nodes in OpenImages v4, but other common demos
    # like "bird" are not).
    with open(ontology_file, "r") as f:
        hierarchy = json.load(f)
    return {
        name: get_all_subclasses(class_root_ids[name], hierarchy) for name in class_root_ids
    }


def list_images_containing(
    class_id_sets: Mapping[str, Set[str]],
    annotations: str,
    n_per_class: int,
    skip_images: Set[str]=set(),
    n_offset: int=0
) -> Set[str]:
    """Deterministic linear search for Open Images matching class_id_sets
    
    Collect `n_per_class` example images for each label, starting at the `n_offset`-th matching
    image and searching deterministically for consistent batch results.
    
    Parameters
    ----------
    class_id_sets : Mapping[str, Set[str]]
        Mapping from our class labels to sets of Open Images class IDs
    annotations : str
        Location of Open Images (bounding boxes) annotations file
    n_per_class : int
        Number of images to retrieve for each label
    skip_images : Set[str] = {}
        Optional set of known-bad-quality image IDs to skip
    n_offset : int
        Number of matching images to skip per class label before returning results
    """
    # Dict[class_name][img_id] -> [class_name, xmin, xmax, ymin, ymax]
    class_bbs = { name: defaultdict(list) for name in class_id_sets }

    n_images_needed = n_per_class + n_offset

    unfilled_class_names = set(class_id_sets.keys())
    with open(annotations, "r") as f:
        for row in csv.reader(f):
            img_id, _, cls_id, conf, xmin, xmax, ymin, ymax, *_ = row
            if (img_id in skip_images):
                continue
            curr_unfilled_class_names = unfilled_class_names.copy()
            for name in curr_unfilled_class_names:
                if (cls_id in class_id_sets[name]):
                    class_bbs[name][img_id].append([name, xmin, xmax, ymin, ymax])
                    if (len(class_bbs[name]) >= n_images_needed):
                        unfilled_class_names.remove(name)

    if (len(unfilled_class_names)):
        warnings.warn(
            "WARNING: Found fewer than ("
            + f"{n_per_class}+{n_offset}={n_images_needed}"
            + ") requested images for the following classes:\n"
            + "\n".join([f"{name} ({len(class_bbs[name])} images)" for name in unfilled_class_names])
        )

    bbs = defaultdict(list)
    for class_name in class_bbs:
        # Take last n_per_class images from each class (for n_offset)
        class_bbs_all_unfiltered = list(class_bbs[class_name].items())
        class_bbs_batch = class_bbs_all_unfiltered[-n_per_class:]
        class_bbs[class_name] = defaultdict(list, class_bbs_batch)
        # Concatenate each class together into the overall `bbs` set
        for (img_id, boxes) in class_bbs_batch:
            bbs[img_id] = bbs[img_id] + boxes

    return set(bbs.keys())
