# app/services/recommend_service.py
from typing import Dict, Any, List
from pathlib import Path
from urllib.parse import urlparse
import logging
from .pipeline import run_pipeline_from_bytes
from ..adapters.chroma_adapter import ChromaAdapter
from ..adapters.s3_adapter import S3Adapter
from ..schemas.recschema import RecommendRequest, RecommendResponse
from botocore.exceptions import ClientError

from ..exception.errors import (
    InvalidImageUrl, UnsupportedCategory,
    S3KeyNotFound, S3FetchError,
    PipelineError, ChromaQueryError,
)

# TOP/PANTS 요청 시 어떤 ChromaDB 컬렉션을 쓸지 결정
COLLECTION_MAP = {"TOP": "fashion_items", "PANTS": "fashion_items_pants"}

logger = logging.getLogger("uvicorn.error")


def _aws_error_code(e: ClientError) -> str:
    return str(e.response.get("Error", {}).get("Code", ""))


class RecommendService:
    def __init__(self, chroma: ChromaAdapter, s3: S3Adapter, yolos_rt, fclip_rt):
        self.chroma = chroma
        self.s3 = s3
        self.yolos_rt = yolos_rt
        self.fclip_rt = fclip_rt

    
    def _s3_url_to_key(self, s3_url: str) -> str:
        """ 
        S3 key 추출
        s3://bucket/label_data/TOP/img.jpg -> label_data/TOP/img.jpg
        """
        if not s3_url.startswith("s3://"):
            raise InvalidImageUrl(f"image_url must be s3://... got={s3_url}")
        u = urlparse(s3_url)
        return u.path.lstrip("/")

    
    def _filename_stem_from_s3_url(self, s3_url: str) -> str:
        """
        s3://...jpg -> 01_sou_000003_000011_front_01outer_01coat_woman
        """
        return Path(self._s3_url_to_key(s3_url)).stem

    def get_collection_name(self, category: str) -> str:
        return COLLECTION_MAP.get(category.upper(), "fashion_items")
    
    def _get_s3_bytes(self, key: str) -> bytes:
        try:
            return self.s3.get_bytes(key)
        except ClientError as e:
            code = _aws_error_code(e)
            if code in ("NoSuchKey", "404", "NotFound"):
                raise S3KeyNotFound(key)
            raise S3FetchError(key, reason=f"ClientError:{code}")
        except Exception as e:
            raise S3FetchError(key, reason=repr(e))

    def recommend(self, image_url: str, category: str, topk: int = 4) -> Dict[str, Any]:

        if category.upper() not in ["TOP", "PANTS"]:
            raise UnsupportedCategory(category)
        # 1) S3 URL 파싱
        img_key = self._s3_url_to_key(image_url)
        img_bytes = self.s3.get_bytes(img_key)
        query_stem = self._filename_stem_from_s3_url(image_url)

        # 2) 파이프라인: crop → embedding
        try:
            out = run_pipeline_from_bytes(
                img_bytes=img_bytes,
                category=category,
                yolos_rt=self.yolos_rt,
                fclip_rt=self.fclip_rt,
                score_thresh=0.3,
            )
        except Exception as e:
            raise PipelineError(reason=repr(e))
        
        if out is None:
            return {"query": {"image_url": image_url, "category": category.upper()}, "product_ids": []}

        # 3) Chroma 검색
        col_name = self.get_collection_name(category)

        try:
            results = self.chroma.query(
                collection_name=col_name,
                query_embedding=out["embedding"].numpy().tolist(),
                n_results=topk + 1, # topk+1 (자기자신 제외용)
                where={"category": category.upper()}, # 카테고리 필터링
            )
        except Exception as e:
            raise ChromaQueryError(collection=col_name, reason=repr(e))
        
        metadatas = (results.get("metadatas") or [[]])[0] or []

        # 4) product_id 추출
        product_ids: List[str] = []
        for meta in metadatas:
            if len(product_ids) >= topk:
                break
            
            # 자기 자신 제외
            if meta.get("image_id") == query_stem:
                continue
            
            # chroma db의 meta 데이터에 있는 label_json_key 사용
            label_json_key = meta.get("label_json_key")
            if not label_json_key:
                print(f"[SKIP] No label_json_key for {meta.get('image_id')}")
                continue
            
            try:
                json_data = self.s3.get_product_info(label_json_key) #JSON 내용 가져오기
                product_id = json_data.get("product_id")
                if product_id:
                    product_ids.append(str(product_id))
                    print(f"[HIT] {label_json_key} → {product_id}")
            except Exception as e:
                print(f"[MISS] {label_json_key}: {e}")
                continue

        return {
            "query": {
                "image_url": image_url,
                "image_key": img_key,
                "category": category.upper(),
                "query_stem": query_stem,
            },
            "product_ids": product_ids,
        }

    def recommend_many(self, reqs: List[RecommendRequest], topk: int = 4) -> List[RecommendResponse]:
        """배치 추천: 여러 이미지 동시에 처리"""
        out: List[RecommendResponse] = []
        for r in reqs:
            one = self.recommend(r.image_url, r.category, topk=topk)
            out.append(
                RecommendResponse(
                    image_url=r.image_url,
                    category=r.category.upper(),
                    product_ids=one.get("product_ids", []),
                )
            )
        return out
