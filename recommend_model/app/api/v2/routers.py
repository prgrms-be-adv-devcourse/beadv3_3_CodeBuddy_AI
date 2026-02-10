# 라우터 = 큰 FastAPI 앱을 작은 조각들로 나누는 도구
# 1초에 100개 API를 한 파일에 쓰면 읽기 힘들고 수정이 어렵기 때문에 API를 여러 개로 나누어서 처리하는 작업마다 API를 다르게 놔주도록 해줍니다. 그 API를 미니로 나누도록 해주는 게 미니 라우터입니다.

from fastapi import APIRouter
from .endpoints.recommend import router as recommend_router

api_router = APIRouter() # V2 API 전체 담당 미니 Fast API
api_router.include_router(recommend_router, tags=["recommend"]) #추천 라우터를 V2에 포함.
