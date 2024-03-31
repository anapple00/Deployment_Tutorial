import traceback

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse


# When an exception is raised, it will be caught by the exception handler
async def internal_server_error_handler(request: Request, exc: Exception):
    tb = traceback.format_exc(-3)  # Only record the last 3 lines of the traceback
    _error_info = {
        "tb": tb,  # locate the error
        "ClassName": type(exc).__name__,  # or "exc.__class__.__name__"
        "content": str(exc),
    }
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder({"detail": _error_info}),
        headers={"X-Request-Id": "There goes my error."}
    )


# When there is no corresponding api name, it will be caught by the exception handler
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": exc.detail},
    )


# When the format of the request is incorrect, it will be caught by the exception handler
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body})
    )
