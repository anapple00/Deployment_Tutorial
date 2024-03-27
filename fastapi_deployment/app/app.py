from fastapi import FastAPI

middleware = []
app = FastAPI(
    title="AI Model Deployment",
    middleware=middleware,
)


def register_router(_app):
    from fastapi_deployment.app.api.named_entity_recognition_api import router as ner_router
    from fastapi_deployment.app.api.text_classification_api import router as text_classification_router
    from fastapi_deployment.app.api.seq2seq_api import router as seg2seg_router
    _app.include_router(ner_router, prefix="/api/test")  # 前缀必须以下划线开始，不能以下划线结束
    app.include_router(text_classification_router, prefix="/api/test")
    _app.include_router(seg2seg_router, prefix="/api/test")


register_router(app)
