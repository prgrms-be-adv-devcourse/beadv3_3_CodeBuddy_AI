from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .errors import AppError


def install_exception_handlers(app: FastAPI) -> None:

    # 서비스 로직에서 raise AppError(...) 또는 raise InvalidImageUrl(...)를 던지면, FastAPI가 호출하는 핸들러
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code, # HTTP 상태코드
            content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail or {}}},
        )


    """
    검토할 메서드 2개 -> PR 참고
    """
    # FastAPI/Starlette 내부에서 발생하는 HTTPException을 잡는 핸들러
    # ex) 404 Not Found, 401 Unauthorized 등
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_EXCEPTION", "message": exc.detail, "detail": {}}},
        )

    # 클라이언트가 요청 바디/쿼리/경로 파라미터를 잘못 보내서 Pydantic 검증이 실패할 때 FastAPI가 호출하는 핸들러
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "VALIDATION_ERROR", "message": "Invalid request.", "detail": {"errors": exc.errors()}}},
        )