# app/errors.py  (원하면 main.py에 같이 둬도 됨)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AppError(Exception):
    code: str
    status_code: int
    message: str
    detail: Optional[Dict[str, Any]] = None


class InvalidImageUrl(AppError):
    def __init__(self, image_url: str):
        super().__init__("INVALID_IMAGE_URL", 400, "image_url must start with s3://", {"image_url": image_url})

class InvalidTopK(AppError):
    def __init__(self, topk: int):
        super().__init__("INVALID_TOPK", 400, "topk must be between 1 and 50.", {"topk": topk})

class S3KeyNotFound(AppError):
    def __init__(self, key: str):
        super().__init__("S3_KEY_NOT_FOUND", 404, "S3 object not found.", {"key": key})

class S3AccessDenied(AppError):
    def __init__(self, key: str):
        super().__init__("S3_ACCESS_DENIED", 403, "S3 access denied.", {"key": key})

class S3FetchError(AppError):
    def __init__(self, key: str, reason: str):
        super().__init__("S3_FETCH_ERROR", 502, "Failed to fetch object from S3.", {"key": key, "reason": reason})

class PipelineError(AppError):
    def __init__(self, reason: str):
        super().__init__("PIPELINE_ERROR", 503, "Failed to run image pipeline.", {"reason": reason})

class ChromaQueryError(AppError):
    def __init__(self, collection: str, reason: str):
        super().__init__("CHROMA_QUERY_ERROR", 502, "Vector DB query failed.", {"collection": collection, "reason": reason})