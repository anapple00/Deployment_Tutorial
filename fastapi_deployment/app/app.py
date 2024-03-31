from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

middleware = []
app = FastAPI(
    title="AI Model Deployment",
    middleware=middleware,
)


def register_router(_app):
    from api.named_entity_recognition_api import router as ner_router
    from api.text_classification_api import router as text_classification_router
    from api.seq2seq_api import router as seg2seg_router
    _app.include_router(ner_router, prefix="/api/test")  # 前缀必须以下划线开始，不能以下划线结束
    app.include_router(text_classification_router, prefix="/api/test")
    _app.include_router(seg2seg_router, prefix="/api/test")


def register_exception_handler(_app):
    from exception import (internal_server_error_handler,
                           http_exception_handler,
                           request_validation_exception_handler,
                           )
    _app.add_exception_handler(Exception, internal_server_error_handler)
    _app.add_exception_handler(HTTPException, http_exception_handler)
    # _app.add_exception_handler(ApiException, api_exception_handler)
    _app.add_exception_handler(RequestValidationError, request_validation_exception_handler)


register_router(app)
register_exception_handler(app)
