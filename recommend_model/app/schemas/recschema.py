from pydantic import BaseModel, Field
from typing import List

class RecommendRequest(BaseModel):
    # 로컬 경로(컨테이너 기준) 또는 파일명(상대경로)을 받는다고 명시
    image_url: str = Field(..., description="Local path under RAW_ROOT or absolute path")
    category: str = Field(..., description="TOP | PANTS")

class RecommendResponse(BaseModel):
    image_url: str
    category: str
    product_ids: List[str]
