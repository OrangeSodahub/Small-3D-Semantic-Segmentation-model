"""
various helper functions to make life easier
"""

import time
import subprocess
import os
from pathlib import Path
import sys
from datetime import datetime
import logging
import re
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def setup_logger():
    """
    setup the logging mechanism where log messages are saved in time-encoded txt files as well as to the terminal
    :return: directory path in which logs are saved
    """
    directory_path = f"logs/{datetime.now():%Y-%m-%d@%H:%M:%S}_{id_generator()}"

    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(pathname)s:%(lineno)04d] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f"{directory_path}/{datetime.now():%Y-%m-%d@%H:%M:%S}_{id_generator()}.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logging.root = logger

    logging.info('START LOGGING')
    logging.info(f"Current Git Version: {git_version()}")

    return directory_path

def git_version():
    """
    return git revision such that it can be also logged to keep track of results
    :return: git revision hash
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision