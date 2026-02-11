# app/services/recommend_service.py
from typing import Dict, Any, List
from pathlib import Path
from urllib.parse import urlparse

from .pipeline import run_pipeline_from_bytes
from ..adapters.chroma_adapter import ChromaAdapter
from ..adapters.s3_adapter import S3Adapter
from app.schemas.recschema import RecommendRequest, RecommendResponse

COLLECTION_MAP = {"TOP": "fashion_items", "PANTS": "fashion_items_pants"}


class RecommendService:
    def __init__(self, chroma: ChromaAdapter, s3: S3Adapter, yolos_rt, fclip_rt):
        self.chroma = chroma
        self.s3 = s3
        self.yolos_rt = yolos_rt
        self.fclip_rt = fclip_rt

    def _s3_url_to_key(self, s3_url: str) -> str:
        """s3://bucket/a/b/c.jpg -> a/b/c.jpg"""
        if not s3_url.startswith("s3://"):
            raise ValueError(f"image_url must be s3://... got={s3_url}")
        u = urlparse(s3_url)
        return u.path.lstrip("/")

    def _filename_stem_from_s3_url(self, s3_url: str) -> str:
        """s3://...jpg -> 01_sou_000003_000011_front_01outer_01coat_woman"""
        return Path(self._s3_url_to_key(s3_url)).stem

    def get_collection_name(self, category: str) -> str:
        return COLLECTION_MAP.get(category.upper(), "fashion_items")

    def recommend(self, image_url: str, category: str, topk: int = 4) -> Dict[str, Any]:
        # 1) 쿼리 이미지 다운로드
        img_key = self._s3_url_to_key(image_url)
        img_bytes = self.s3.get_bytes(img_key)
        query_stem = self._filename_stem_from_s3_url(image_url)

        # 2) 파이프라인: crop → embedding
        out = run_pipeline_from_bytes(
            img_bytes=img_bytes,
            category=category,
            yolos_rt=self.yolos_rt,
            fclip_rt=self.fclip_rt,
            score_thresh=0.3,
        )
        if out is None:
            return {
                "query": {"image_url": image_url, "category": category.upper()},
                "product_ids": [],
            }

        # 3) Chroma 검색
        col_name = self.get_collection_name(category)
        results = self.chroma.query(
            collection_name=col_name,
            query_embedding=out["embedding"].numpy().tolist(),
            n_results=topk + 1,
            where={"category": category.upper()},
        )
        metadatas = (results.get("metadatas") or [[]])[0] or []

        # 4) ★Chroma 메타의 label_json_key 직접 사용★ (S3 검색 0!)
        product_ids: List[str] = []
        for meta in metadatas:
            if len(product_ids) >= topk:
                break
            
            # 자기 자신 제외
            if meta.get("image_id") == query_stem:
                continue
            
            # ★label_json_key 바로 사용★
            label_json_key = meta.get("label_json_key")
            if not label_json_key:
                print(f"[SKIP] No label_json_key for {meta.get('image_id')}")
                continue
            
            try:
                json_data = self.s3.get_product_info(label_json_key)
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
