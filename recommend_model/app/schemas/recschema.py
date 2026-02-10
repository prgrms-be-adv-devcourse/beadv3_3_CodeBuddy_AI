# Java ↔ FastAPI 간 “데이터 계약”을 명확히 합니다.

# 자동 문서(Swagger) 생성에도 반영됩니다.

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class RecommendRequest(BaseModel):
    image_url: str = Field(..., description="s3://bucket/key | http(s)://... | file:/data/... | /data/...")
    category: str = Field(..., description="TOP | PANTS")
    topk: int = Field(4, ge=1, le=50)

class RecommendItem(BaseModel):
    product_id: str
    similarity: float
    image_url: str
    category: str

class RecommendResponse(BaseModel):
    query: Dict[str, Any]
    recommendations: List[RecommendItem]
