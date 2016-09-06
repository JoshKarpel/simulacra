import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Potential:
    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, **kwargs):
        raise NotImplementedError
