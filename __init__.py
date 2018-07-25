# import the submodules
from . import kern
from . import models
from . import tensors
from . import linalg
from . import grid

# create a function which allows for setup of debugging
def debug():
    """
    Sets up the package for debugging. This includes:
        * remove existing logging handlers and sets the handler level to logging.DEBUG
    """
    # First, remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    # now respecify the logging handlers
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                        datefmt='[ %H:%M:%S ]')


__all__ = filter(lambda s:not s.startswith('_'), dir())
__version__ = '1.0'

# set up logging (note if a handler has already been set then this won't do anything)
import logging
import sys
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='[ %H:%M:%S ]')
