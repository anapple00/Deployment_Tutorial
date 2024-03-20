import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Demo", middleware=[])


class InputSchema(BaseModel):
    string: str


class OutputSchema(BaseModel):
    string: str


@app.post("/test", response_model=OutputSchema)
def test_fastapi(request: InputSchema):
    return OutputSchema(string=f"get {request.string}, test successful.")


def main(_app):
    # log_config = {
    #     "version": 1,
    #     "disable_existing_logger": False,
    #     "root": False,
    # }
    # uvicorn.run(_app, host="0.0.0.0", port=6006, reload=False, log_config=log_config)
    uvicorn.run(_app, host="0.0.0.0", port=6005, reload=False)


if __name__ == "__main__":
    main(app)
