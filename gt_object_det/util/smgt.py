"""Utilities for working with SageMaker (Ground Truth) Manifest Files

(Augmented) Manifest Files are simply text files in JSON Lines format: Each line is a valid JSON
object. Specifically, each line is a JSON *object*, where:

- The value of any top-level attribute ending with "-ref" is a S3 URI string referring to an
  object; e.g. "s3://bucket-name/path-to-my-object"...
- Typically, the file describes an annotated dataset where for each record "source-ref" is a data
  file (such as an image), and some other attribute (e.g. "label") is a (perhaps complex 
  structured) annotation.
- Additional attributes may be added to specify further metadata, such as if multiple labelling
  jobs have been performed on each data object.

These utilities simplify importing manifests (updating "-ref"s to new buckets, etc); and merging
multiple manifest files into one.

See Also
--------

- http://jsonlines.org/
- https://docs.aws.amazon.com/sagemaker/latest/dg/augmented-manifest.html
"""

# Python Built-Ins:
import logging
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

# External Dependencies
import boto3
import json
import numpy as np

logger = logging.getLogger("util.openimages")
logger.setLevel(logging.INFO) # Feel free to change to logging.DEBUG to see more detailed output


def s3_uri_to_bucket_and_key(uri: str) -> Tuple[str, str]:
    """E.g. 's3://bucket/my/obj/uri' -> 'bucket', 'my/obj/uri'"""
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def translate_manifest_refs(
    source_manifest: str,
    target_manifest: str,
    translator_fn: Callable[[str], str],
) -> None:
    """Apply `translator_fn` to translate the -ref fields in a manifest file
    
    Record fields ending in "-ref" are references to an S3 location. This iterates through all
    records in a file and all "-ref" fields, calling `translator_fn` to update them.
    
    Parameters
    ----------
    source_manifest : str
        Local file name/path of existing manifest file to take as input
    target_manifest: str
        Local file name/path of new manifest file to write out to
    translator_fn: Callable[[str], str]
        Function to translate a ref in source manifest to the target (e.g. by moving to new bucket)
    """
    with open(source_manifest, "r") as f_source:
        with open(target_manifest, "w") as f_target:
            for line in f_source:
                datum = json.loads(line)
                for key in filter(lambda k: k.endswith("-ref"), datum.keys()):
                    datum[key] = translator_fn(datum[key])
                f_target.write(json.dumps(datum) + "\n")


def merge_manifests(
    target_field: str,
    *args: Dict,
    shuffle: bool=False,
) -> Iterable[Any]:
    """Merge manifest files and return parsed merged data
    
    Consolidate a set of manifest files with different label attribute names into a merged data
    set with single common attribute name.
    
    Parameters
    ----------
    target_field : str
        The field name on the merged manifest where annotation data should be stored
    *args : Dict
        With attributes `file` (an input manifest file path) and `field` (the field to fetch
        annotations from in each file)
    shuffle : bool=False
        By default, manifests will be concatenated in provided order. Set True for (numpy-based)
        random shuffle.
    """
    result = []
    for source in args:
        logger.info(f"Merging {source['file']}")
        with open(source["file"], "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            # Standardize label names:
            for datum in data:
                datum[target_field] = datum.pop(source["field"])
                if (source["field"] + "-metadata") in datum:
                    datum[target_field + "-metadata"] = datum.pop(source["field"] + "-metadata", None)
            result += data
    if shuffle:
        np.random.shuffle(result)
    return result


class ManifestRefImporter:
    """A callable that copies a referenced resource to target bucket and returns target location
    
    If you have a manifest file of annotations originally generated for another account or region,
    you may need to copy the referenced objects into your own bucket to train a model.
    
    Assuming you have access to the objects (either in the original source bucket, or in a third
    "repository" bucket), this class helps by creating a Callable[[str], str] which will copy a
    referenced object into your target bucket - and return the URI of the new copy.
    
    Parameters
    ----------
    source : Callable[[str], str]
        A callable generating the active/subtree part of the filename from the full reference in
        the source manifest.
    target : str
        An S3 URI denoting the "base" portion of target refs: i.e. the target bucket and folder 
        where objects will be copied to.
    repository: Optional[str]=None
        A separate S3 URI denoting the bucket and folder which objects can be copied **from** -
        otherwise the object URI listed in the source manifest file will be used.
    session: Optional[boto3.session.Session]
        An existing boto3 session to use, otherwise a new one will be generated on __init__
    
    Examples
    --------
    
    My colleague's manifest file references images in S3 that I don't have access to, such as:
    "s3://their-bucket/their/folder/251d4c429f6f9c39.jpg". I don't need their folder structure,
    but know that the filename is always matching the published OpenImages (test) dataset.
    
    To use their annotations in a training job on my account, I want to upload the referenced
    images to s3://my-bucket-in-the-right-region/images/

    >>> import_fn = ManifestRefImporter(
    >>>     lambda s: s.rpartition("/")[2],
    >>>     "s3://my-bucket-in-the-right-region/images/",
    >>>     repository="s3://open-images-dataset/test/",
    >>>     session=session,
    >>> )
    """
    def __init__(
        self,
        source:Callable[[str], str],
        target:str,
        repository:Optional[str]=None,
        session:Optional[boto3.session.Session]=None
    ):
        """Please see class docstring"""
        self._source = source
        
        if target.lower().startswith("s3://"):
            self._target_bucket_name, self._target_prefix = s3_uri_to_bucket_and_key(target)
        else:
            raise ValueError("`target` must be a qualified s3:// folder URI")
        if not self._target_prefix.endswith("/"):
            self._target_prefix += "/"

        if session is None:
            session = boto3.session.Session()
        self._target_bucket = session.resource("s3").Bucket(self._target_bucket_name)
        
        if repository:
            self._repo_bucket_name, self._repo_prefix = s3_uri_to_bucket_and_key(repository)
            if not self._repo_prefix.endswith("/"):
                self._repo_prefix += "/"
        else:
            # We'll just try to copy from the object URI as mentioned in the source manifest file
            self._repo_bucket_name = None
            self._repo_prefix = None

    def __call__(self, source_ref:str) -> str:
        """Copy the object referenced by `source_ref` into the target bucket and return new URI
        """
        filename = self._source(source_ref)
        if self._repo_bucket_name and self._repo_prefix:
            # A separate repository 
            self._target_bucket.copy(
                {
                    "Bucket": self._repo_bucket_name,
                    "Key": self._repo_prefix + filename
                },
                self._target_prefix + filename
            )
        else:
            obj_bucket, obj_key = s3_uri_to_bucket_and_key(source_ref)
            self._target_bucket.copy(
                {
                    "Bucket": obj_bucket,
                    "Key": obj_key
                },
                self._target_prefix + filename
            )
        return f"s3://{self._target_bucket_name}/{self._target_prefix}{filename}"
