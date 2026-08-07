import logging
from importlib.metadata import version

name = "spliceai"
__version__ = version(name)

logger = logging.getLogger(name)
