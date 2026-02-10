from typing import Dict, Any, Optional
from pathlib import Path
import os, json

from .pipeline import run_pipeline_from_bytes
from ..adapters.chroma_adapter import ChromaAdapter

COLLECTION_MAP = {"TOP": "fashion_items", "PANTS": "fashion_items_pants"}

class RecommendService:
    def __init__(self, chroma: ChromaAdapter, yolos_rt, fclip_rt):
        self.chroma = chroma
        self.yolos_rt = yolos_rt
        self.fclip_rt = fclip_rt

        self.raw_root = Path(os.getenv("RAW_ROOT", "/tmp/raw"))
        self.label_root = Path(os.getenv("LABEL_ROOT", "/tmp/label"))

    def _get_image_bytes(self, image_url: str) -> bytes:
        """로컬 파일만 처리"""
        img_path = self.raw_root / image_url  # 상대경로 합치기
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img_path.read_bytes()

    def get_collection_name(self, category: str) -> str:
        return COLLECTION_MAP.get(category.upper(), "fashion_items")

    def resolve_raw_image_path(self, image_url: str) -> Path:
        """
        image_url이 절대경로(/data/...)면 그대로 사용,
        상대경로면 RAW_ROOT 기준으로 합침.
        """
        p = Path(image_url)
        if p.is_absolute():
            return p
        return self.raw_root / image_url

    def find_json_by_filename(self, img_id: str) -> Optional[dict]:
        """이미지 ID(이미지 파일 이름)로 JSON 파일 찾기 (.json 확장자 자동 추가)"""
        json_filename = f"{img_id}.json"  # ← 이거 추가!
        print(f"[DEBUG] Searching for JSON: {json_filename}")
        
        for json_path in self.label_root.rglob(json_filename):
            print(f"[DEBUG] Found JSON: {json_path}")
            try:
                return json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[DEBUG] Failed to read {json_path}: {e}")
                continue
        
        print(f"[DEBUG] JSON not found: {json_filename}")
        print(f"[DEBUG] LABEL_ROOT sample files: {[f.name for f in list(self.label_root.rglob('*.json'))[:3]]}")
        return None

    def recommend(self, image_url: str, category: str, topk: int = 4) -> Dict[str, Any]:
        # 1) 로컬 이미지 bytes 로드
        img_path = self.resolve_raw_image_path(image_url)
        img_bytes = img_path.read_bytes()
        
        # 쿼리 이미지의 파일명(확장자 제외) 추출 (자기 자신 제외용)
        query_img_id = Path(img_path).stem 

        # 2) bytes -> crop -> embedding
        out = run_pipeline_from_bytes(
            img_bytes=img_bytes,
            category=category,
            yolos_rt=self.yolos_rt,
            fclip_rt=self.fclip_rt,
            score_thresh=0.3,
        )
        if out is None:
            return {
                "query": {"image_path": str(img_path), "category": category.upper(), "collection_used": self.get_collection_name(category)},
                "product_ids": [],
            }

        # 3) Chroma 검색
        query_emb = out["embedding"].numpy().tolist()
        col_name = self.get_collection_name(category)

        # [수정 포인트 1] 자기 자신이 포함될 것을 대비해 topk + 1개를 요청합니다.
        search_limit = topk + 1

        results = self.chroma.query(
            collection_name=col_name,
            query_embedding=query_emb,
            n_results=search_limit,  # topk 대신 search_limit 사용
            where={"category": category.upper()},
        )

        ids = (results.get("ids") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []

        product_ids: list[str] = []
        filtered_neighbor_ids: list[str] = [] # 결과 JSON의 neighbors에 넣을 ID 리스트

        for i, meta in enumerate(metadatas):
            # 목표 개수(topk)를 채웠다면 중단
            if len(product_ids) >= topk:
                break

            if not meta or 'image_id' not in meta:
                continue
            
            img_id = meta['image_id']
            
            # [수정 포인트 2] 자기 자신이면 건너뜀
            if img_id == query_img_id:
                print(f"[DEBUG {i}] Skipping self: {img_id}")
                continue
            
            # JSON 정보 조회
            label_json = self.find_json_by_filename(img_id)
            if label_json and label_json.get("product_id"):
                product_ids.append(str(label_json["product_id"]))
                filtered_neighbor_ids.append(ids[i]) # 자기 자신이 아닌 이웃 ID만 추가
                print(f"[DEBUG {i}] SUCCESS: {label_json['product_id']}")

        return {
            # [수정 포인트 3] neighbors에도 필터링된 리스트 전달
            "query": {
                "image_path": str(img_path), 
                "category": category.upper(), 
                "collection_used": col_name, 
                "neighbors": filtered_neighbor_ids
            },
            "product_ids": product_ids,
        }