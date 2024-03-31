from pydantic import BaseModel, Extra


class InputSchema(BaseModel, extra=Extra.forbid):
    query: str
    model_type: str
    task: str
    dataset: str


class OutputSchema(BaseModel, extra=Extra.forbid):
    response: str
