import time

from loguru import logger


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {round(end_time - start_time)} seconds to execute.")
        return result

    return wrapper
