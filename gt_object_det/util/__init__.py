# Python Built-Ins:
import logging
import sys

# Fix logging for Jupyter notebooks:
logging.basicConfig()
root_logger = logging.getLogger()
root_logger.handlers[0].stream = sys.stdout

from .plotting import *

from . import openimages
from . import smgt
