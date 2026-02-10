from fastapi import APIRouter, Depends, Request
from app.schemas.recschema import RecommendRequest, RecommendResponse
from app.services.recommend_service import RecommendService

router = APIRouter()

def get_service(request: Request) -> RecommendService:
    return request.app.state.recommend_service

@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, service: RecommendService = Depends(get_service)):
    return service.recommend(req.image_url, req.category)
