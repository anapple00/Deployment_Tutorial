import os

from loguru import logger
from pydantic import BaseModel

_EXCLUDE = ["AWS_REGION_NAME", "ENV", "SECRET_NAME"]


class BaseConfig(BaseModel):
    """
    config load priority:
        local_setting >secret manager > env > default
    """

    @classmethod
    def load(cls, **local_settings):
        config = {}

        for k, v in cls.model_fields.items():
            if k in local_settings.keys():
                config[k] = local_settings[k]
                logger.info(f"[Config] load [{k}] : {local_settings[k]}")

        return cls(**config)
