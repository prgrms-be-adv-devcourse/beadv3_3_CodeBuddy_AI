from pydantic import BaseModel, Field
from typing import List, Dict, Any

class RecommendRequest(BaseModel):
    # 로컬 경로(컨테이너 기준) 또는 파일명(상대경로)을 받는다고 명시
    image_url: str = Field(..., description="Local path under RAW_ROOT. e.g. subdir/img.jpg or /data/.../img.jpg")
    category: str = Field(..., description="TOP | PANTS")

class RecommendResponse(BaseModel):
    product_ids: List[str]
