# Semantic Segmentation

So you've run through the [official example notebook](https://github.com/aws/amazon-sagemaker-examples/tree/master/introduction_to_amazon_algorithms/semantic_segmentation_pascalvoc) and got to grips with the [Amazon SageMaker Semantic Segmentation algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html) - great!

You might next be wondering:

> I don't want to mess around with image processing in my front end app... Can I set up a Lambda function in front of the SageMaker model, which post-processes the results into a nice human-viewable image?

The answer is yes, and here we provide an example to get you started. 👍


## Pre-Requisites

We assume you already have:

- A Semantic Segmentation model deployed to a **SageMaker Endpoint** (see the notebook linked above if not), and that you know your endpoint's name (check the [SageMaker Console](https://console.aws.amazon.com/sagemaker/home?#/endpoints)).
- Access to deploy CloudFormation stacks (including Lambda and IAM resources) into the AWS account you're working in.
- [Docker](https://www.docker.com/products/docker-desktop) and the [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html) installed in the environment you're working on.


## Design

This example uses the [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html) to build a Lambda function in a **containerized environment**, which is helpful for ensuring the required libraries are bundled in a format compatible with the AWS Lambda Python runtime: All you need to do is specify libraries in the `requirements.txt` file.

The SAM template [template.sam.yaml](template.sam.yaml) simply creates, configures and permissions a Lambda function.

### Interface

For full details of the interface exposed, see the `handler()` doc in [fn-sseg/main.py](fn-sseg/main.py).

This function renders (typically translucent) human-viewable PNG images for a wide variety of semantic segmentation visualization use-cases, including:

- Single-color mask with opacity representing the confidence of detecting the particular supplied `TargetClassId` at each pixel.
- Single-color mask with fixed opacity where `TargetClassId` was the most confidently predicted class, and transparency in other regions.
- Color-mapping the detected class ID per pixel
- Color-mapping the confidence of detecting `TargetClassId` at each pixel

**For example:**

```python
import base64
import io
import json
import boto3
from matplotlib import pyplot as plt
from PIL import Image

lambdaclient = boto3.client("lambda")

with open("test.jpg", "rb") as imfile:
    imbytes = imfile.read()

response = lambdaclient.invoke(
    FunctionName="sseg-lambda-FunctionCallModel-FGTEOGVD6NMF",
    Payload=json.dumps({
        # Provide base64 image data or S3 URI:
        "ImageBase64": base64.b64encode(imbytes).decode("utf-8"),
        #"ImageS3Uri": "s3://DOC-EXAMPLE-BUCKET/my/image.jpg',
        # Specify a target class ID, unless colormapping all:
        "TargetClassId": 14,
        # Configurable max opacity, so you can overlay on input image:
        "Alpha": 0.8,
        # Analyze winning classes only, or full confidence-per-class-per-pixel:
        "Probabilistic": True,
        # Configure single-color mask, or matplotlib colormaps:
        #"MaskColor": [0.0, 1.0, 1.0]
        #"MaskColorMap": "jet"
    }),
)

# JSON response:
result = json.loads(response["Payload"].read())

# Plot in Python:

plt.imshow(Image.open(io.BytesIO(base64.b64decode(result["MaskBase64"]))))
plt.show()

# Or display with HTML, if you're in a JS environment rather than Python:
display(HTML(
    f"""<img src=data:image/png;base64,{result["MaskBase64"]}" style="background-image: url('source-image.jpg');"/>"""
))
```

### Notes & Caveats

- **Base64 encoding** is used (rather than the more efficient [Base85 AKA Ascii85](https://en.wikipedia.org/wiki/Ascii85)) for inline request and response image data, to facilitate easier displaying in the browser for web app use cases.
- Lambda function request and response payloads are JSON-serializable (hence the encoding requirement) and **[limited to 6MB](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)**, so consider S3-based patterns rather than inline if dealing with large images - and remember that Base64 will inflate image sizes over the raw JPEG/PNG files.
- This Lambda function operates on **pixel arrays** to transform the raw model outputs into human-viewable images, and so can be both memory and CPU-hungry. Be aware that some super **high-resolution use cases may exceed the [maximum configurable Lambda memory allocation](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)**. Different use-cases (e.g. probabilistic vs non-probabilistic) may have different resource requirements. See configuration notes in [template.sam.yaml](template.sam.yaml).
- Since [Apache MXNet](https://mxnet.apache.org/versions/1.7.0/) would not fit in the Lambda bundle size and the [PyPI recordio package](https://pypi.org/project/recordio/) is "Archived" on GitHub, we implement a kind-of-DMLC-compatible **RecordIO reader** in Python, in [fn-sseg/deserializers.py]. It seems to work well enough for this use case, but this approach is not particularly performant or robust!


## Deploying with AWS SAM

**Step 1: Build the Lambda bundle**

The `--use-container` flag instructs SAM to use a Docker container for the build process, ensuring the installed dependencies from [fn-sseg/requirements.txt](fn-sseg/requirements.txt) will be compatible with the Lambda runtime.

```bash
# (Or omit the --profile argument if you don't need to use one)
sam build \
    --use-container \
    --template template.sam.yaml \
    --profile $AWSPROFILE
```

**Step 2: Package the artifacts for CloudFormation**

Once the Lambda bundles have been built, resolve the SAM template into a regular CloudFormation template by staging the Lambda bundles to an S3 bucket. 

```bash
# (Stage your built Lambda bundles to an S3 bucket of your choice)
sam package \
    --output-template-file template.tmp.yaml \
    --s3-bucket $STAGING_S3_BUCKET \
    --s3-prefix $STAGING_S3_PREFIX \
    --profile $AWSPROFILE
```

**Step 3: Deploy the CloudFormation stack**

Deploy your built `template.tmp.yaml` either via CloudFormation console or AWS SAM CLI, specifying a stack name and the name of the SageMaker endpoint you want to configure the function for.

```bash
sam deploy \
    --template-file template.tmp.yaml \
    --stack-name $STACKNAME \
    --profile $AWSPROFILE \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
        SageMakerEndpoint=$SAGEMAKER_ENDPOINT_NAME
```
