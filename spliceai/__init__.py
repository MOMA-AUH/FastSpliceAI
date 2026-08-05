import signal
import sys
from importlib.metadata import version
from logging import getLogger

signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

name = "spliceai"
logger = getLogger(name)
__version__ = version(name)
