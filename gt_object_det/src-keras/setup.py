"""Module setup.py to enable inference container install

The SageMaker TensorFlow v1.12 inference container doesn't have `wheel` installed by default; which causes
an error like this one when our requirements.txt is installed:

https://stackoverflow.com/questions/34819221/why-is-python-setup-py-saying-invalid-command-bdist-wheel-on-travis-ci

...So we supply the below to explicitly call out `wheel` as a pre-dependency
"""

from setuptools import setup

setup(
    name="gt_object_det_keras",
    version="0.1.0",
    # Sometimes setuptools gets sniffy if author is missing:
    author="apac-ml-tfc",
    url="https://github.com/apac-ml-tfc/computer-vision-workshops",
    description="Training and deploying YOLOv3 on SageMaker with tf.keras",
    classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
    python_requires='>=3.5',
    install_requires=[
        # For commentary on dependency versions, see requirements.txt
        "setuptools",
        "kiwisolver<1.2",
        "matplotlib>=2.1.1,<3.1",
        "Pillow>=6.0,<7.0",
    ],
    # matplotlib has some components that need us to build wheels:
    setup_requires=["wheel"],
)
