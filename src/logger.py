"""
Examples
    >>> import logger
    >>> logger.get_logger(__name__)
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="[%(levelname)s] %(name)s: %(message)s")


def get_logger(name):
    return logging.getLogger(name)
