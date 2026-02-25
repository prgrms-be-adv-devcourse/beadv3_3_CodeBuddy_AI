from pydantic import BaseModel
from typing import List, Optional


class RecommendItemPayload(BaseModel):
    imageUrl: str
    category: str

class RecommendRequestPayload(BaseModel):
    requestId: str
    memberId: int
    items: List[RecommendItemPayload]


class RecommendResultItemPayload(BaseModel):
    imageUrl: str
    category: str
    productIds: List[str]

class RecommendResultPayload(BaseModel):
    requestId: str
    memberId: int
    success: bool
    failReason: Optional[str] = None
    result: Optional[List[RecommendResultItemPayload]] = None
