# # 커스텀 예외 정의
# from app.exception.errors import AppError


# # image url이 파싱이 불가능하거나 문제가 있을 때 사용되는 메서드
# class InvalidImageUrl(AppError):
#     def __init__(self, image_url: str):
#         super().__init__(
#             code="INVALID_IMAGE_URL",
#             status_code=400,
#             message="image_url must start with s3://",
#             detail={"image_url": image_url},
#         )

# # category가 TOP/PANTS 지원 목록에 없을 때 사용하는 메서드
# class UnsupportedCategory(AppError):
#     def __init__(self, category: str):
#         super().__init__(
#             code="UNSUPPORTED_CATEGORY",
#             status_code=400,
#             message="Unsupported category.",
#             detail={"category": category},
#         )


# # S3에서 해당 key의 이미지/JSON이 없을 때 사용하는 메서드
# class S3KeyNotFound(AppError):
#     def __init__(self, key: str):
#         super().__init__(
#             code="S3_KEY_NOT_FOUND",
#             status_code=404,
#             message="S3 object not found.",
#             detail={"key": key},
#         )


# #ChromaDB 쿼리 호출 자체가 실패했을 때 사용하는 메서드
# class ChromaQueryError(AppError):
#     def __init__(self, collection: str, reason: str):
#         super().__init__(
#             code="CHROMA_QUERY_ERROR",
#             status_code=502,
#             message="Vector DB query failed.",
#             detail={"collection": collection, "reason": reason},
#         )