import uvicorn


def main(_app):
    from app.app import app
    log_config = {
        "version": 1,
        "disable_existing_logger": False,
        "root": False,
    }
    uvicorn.run(_app, host="0.0.0.0", port=6006, reload=False, log_config=log_config)
    uvicorn.run(_app, host="0.0.0.0", port=6005, reload=False)


if __name__ == "__main__":
    main(app)
