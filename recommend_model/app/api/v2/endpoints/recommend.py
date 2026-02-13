from typing import List
from fastapi import APIRouter, Depends, Request
from ....schemas.recschema import RecommendRequest, RecommendResponse
from ....services.recommend_service import RecommendService

router = APIRouter()

def get_service(request: Request) -> RecommendService:
    return request.app.state.recommend_service

@router.post("/recommend", response_model=List[RecommendResponse])
def recommend(reqs: List[RecommendRequest], service: RecommendService = Depends(get_service)):
    return service.recommend_many(reqs)
