import logging

from vae.config import LOG


_log_format = logging.Formatter("[%(name)s| %(levelname)s]: %(message)s")
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_log_format)
LOG.addHandler(_log_handler)
