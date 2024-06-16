from loguru import logger
from pydantic import BaseModel, Field

from config.config_manager import BaseConfig


class _ENV(BaseModel):
    """load from environment"""
    MONGODB_URL: str = Field(default="")


class _Secret(BaseModel):
    """load from locol_config.py"""

    # AWS config
    AWS_ACCESS_KEY_ID: str = Field(default="")
    AWS_SECRET_ACCESS_KEY: str = Field(default="")
    AWS_DEFAULT_REGION: str = Field(default="")

    # mongoDB config


class Config(BaseConfig, _ENV, _Secret):
    pass


try:
    import local_config  # noqa

    logger.info("local config found, use local config")
    config = Config.load(**{k: v for k, v in local_config.__dict__.items()})

except (ImportError, ModuleNotFoundError):
    config = Config.load()
