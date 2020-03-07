# SageMaker Processing for Video Frame Extraction

Many computer vision applications use video source data, but treat the task as independent analysis of a sequence of images - for latency, resource cost, or project complexity reasons.

There are lots of ways to convert a video to a set of images, but the [Python binding](https://pypi.org/project/opencv-python/) of [OpenCV](https://opencv.org/) is one popular tool for its flexibility: Different projects might need to e.g. extract at different frame rates, crop the results, or combine with other transformations.

AWS also provides a range of differently-optimized environments on which the code might be run, such as:

* As a **Lambda function** (like [this greyscaling example](https://github.com/iandow/opencv_aws_lambda)), which could be triggered to run automatically whenever a new video was uploaded to S3)
* As a **SageMaker Processing Job**, well-suited for on-demand batch processing for data scientists who want scalable compute without leaving the SageMaker workbench.

This example isn't intended as a reference implementation of OpenCV frame extraction logic - but it demonstrates some patterns for using SageMaker Processing that might not be obvious to beginners, like:

* How to use dependency libraries that aren't available in the SageMaker base container images (and especially those like OpenCV that might require extra OS-level libraries beyond a simple `pip install`)
* How to pass parameters in to a processing job (like frame rate, which we pass in but don't implement in the container logic)
* How a processing instance can discover the total number of instances executing the current job, and its place in the list, since this process is a little different than the `SM_HOSTS` and `SM_CURRENT_HOSTS` environment variables often used in SageMaker training jobs.

Note that this use case **doesn't need** to identify instance count (because SageMaker can distribute input data between instances automatically with the [s3_data_distribution_type](https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.ProcessingInput) parameter), but the logic may be useful for other applications e.g. master-slave cluster architectures.
