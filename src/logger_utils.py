# src/logger_utils.py

import logging
import os


def setup_logger(name: str, log_file: str):
    """
    Sets up a named logger that writes to both a file and the console.

    """
    try:
        logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    except Exception as e:
        print(f"Failed to set up logger '{name}' for '{log_file}': {e}")
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)