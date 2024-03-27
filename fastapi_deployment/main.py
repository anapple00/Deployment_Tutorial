import uvicorn


def main():
    from app.app import app
    log_config = {
        "version": 1,
        "disable_existing_logger": False,
        "root": False,
    }
    uvicorn.run(app, host="0.0.0.0", port=6006, reload=False, log_config=log_config)


if __name__ == "__main__":
    main()
