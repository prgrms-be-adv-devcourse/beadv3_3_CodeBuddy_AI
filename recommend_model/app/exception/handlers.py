from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .errors import AppError, UnsupportedCategory,S3KeyNotFound, S3FetchError, PipelineError, ChromaQueryError

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
    
    # UnsupportedCategory 관련 핸들러
    @app.exception_handler(UnsupportedCategory)
    async def unsupported_category_handler(request: Request, exc: UnsupportedCategory):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail
            }}
        )
    

    #S3KeyNotFound 관련 핸들러
    @app.exception_handler(S3KeyNotFound)
    async def s3_key_not_found_handler(request: Request, exc: S3KeyNotFound):
        return JSONResponse(
            status_code=exc.status_code,  # 404
            content={"error": {
                "code": exc.code,           # "S3_KEY_NOT_FOUND"
                "message": exc.message,     # "S3 object not found."
                "detail": exc.detail        # {"key": "image/fake.jpg"}
            }}
        )

    @app.exception_handler(S3FetchError)
    async def s3_fetch_error_handler(request: Request, exc: S3FetchError):
        return JSONResponse(
            status_code=502,  # 또는 500
            content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
        )

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request: Request, exc: PipelineError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "detail": exc.detail or {}
                }
            },
        )
    
    @app.exception_handler(ChromaQueryError)
    async def chroma_query_error_handler(request: Request, exc: ChromaQueryError):
        return JSONResponse(
            status_code=exc.status_code,  # 너는 502로 정의함
            content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail or {}}},
        )