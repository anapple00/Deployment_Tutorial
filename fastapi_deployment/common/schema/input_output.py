from pydantic import BaseModel, Extra, ConfigDict


class InputSchema(BaseModel, extra=Extra.forbid):
    query: str
    model_type: str
    task: str
    dataset: str

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )  # 防止"model_type"报warning


class OutputSchema(BaseModel, extra=Extra.forbid):
    response: str

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )  # 防止"model_type"报warning
