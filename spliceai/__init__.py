import signal
import sys
from importlib.metadata import version

signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

name = "spliceai"
__version__ = version(name)
