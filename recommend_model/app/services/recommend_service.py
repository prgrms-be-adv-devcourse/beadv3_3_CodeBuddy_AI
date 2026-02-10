# 전체 역할: "이미지 URL 입력 → product_id 리스트 출력하는 추천 엔진"
from typing import Dict, Any, Optional


from services.pipeline import run_pipeline_from_bytes
from adapters.chroma_adapter import ChromaAdapter
from adapters.s3_adapter import S3Adapter


#TOP/PANTS 요청 시 어떤 ChromaDB 컬렉션을 쓸지 결정
COLLECTION_MAP = {"TOP": "fashion_items", "PANTS": "fashion_items_pants"}


# 입력: "s3://codebuddy.../image_data/TOP/img.jpg"
# 출력: ("codebuddy...", "image_data/TOP/img.jpg")
def parse_s3_url(s3_url: str) -> tuple[str, str]:
    if not s3_url.startswith("s3://"):
        raise ValueError("Use s3://bucket/key")
    bucket, key = s3_url.replace("s3://", "").split("/", 1)
    return bucket, key


class RecommendService:
    def __init__(self, chroma: ChromaAdapter, s3_adapter: S3Adapter, yolos_rt, fclip_rt):
        self.chroma = chroma # ChromaDB 연결
        self.s3_adapter = s3_adapter # S3 연결
        self.yolos_rt = yolos_rt # YOLO 모델
        self.fclip_rt = fclip_rt # FashionCLIP 모델


    #"TOP" → "fashion_items"
    #"PANTS" → "fashion_items_pants"
    def get_collection_name(self, category: str) -> str:
        return COLLECTION_MAP.get(category.upper(), "fashion_items")


    # 핵심 메서드
    def recommend(self, image_url: str, category: str, topk: int = 4) -> Optional[Dict[str, Any]]:
        # A) 쿼리 이미지 처리
        # 실행 과정:
        # s3://URL → S3 다운로드 → YOLO 크롭 → FashionCLIP 임베딩
        # ↓
        # out = {"embedding": [0.123, ...], "label_name": "coat", "score": 0.85}

        # 1) URL -> bucket/key
        bucket, key = parse_s3_url(image_url)

        # 2) S3에서 이미지 bytes 다운로드
        img_bytes = self.s3_adapter.get_bytes(bucket=bucket, key=key)

        # 3) bytes -> crop -> embedding
        out = run_pipeline_from_bytes(
            img_bytes=img_bytes,
            category=category,
            yolos_rt=self.yolos_rt,
            fclip_rt=self.fclip_rt,
            score_thresh=0.3,
        )
        if out is None:
            return None

        # Chroma DB 벡터 검색 (label_json_key 필터링 가능)
        query_emb = out["embedding"].numpy().tolist()

        col_name = self.get_collection_name(category)
        results = self.chroma.query(
            collection_name=col_name,
            query_embedding=query_emb,
            n_results=topk,
            where={
                "category": category.upper(),
                "label_json_key": {"$exists": True}  # ✅ json_key 있는 문서만
            },
        )

        # 결과
        # {
        #   "ids": [["img1.jpg", "img2.jpg"]],
        #   "distances": [[0.12, 0.15]],
        #   "metadatas": [{"label_json_key": "label_data/TOP/VL_TOP_coat/img1.json"}, ...]
        # }
        ids = (results.get("ids") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []

        # ✅ C) Chroma metadata의 label_json_key 직접 사용 → S3 json 로드 → product_id 추출
        items = []
        for i, (image_key, meta) in enumerate(zip(ids[:topk], metadatas[:topk])):
            # metadata에서 label_json_key 가져오기
            json_key = meta.get("label_json_key")
            
            if not json_key:
                print(f"Warning: No label_json_key for {image_key}")
                continue

            # S3에서 json 로드
            try:
                label_json = self.s3_adapter.get_json(bucket=bucket, key=json_key)
                product_id = label_json.get("product_id")
                
                items.append({
                    "rank": i + 1,
                    "product_id": product_id,
                    "distance": float(distances[i]) if i < len(distances) else None,
                    "image_key": image_key,
                    "json_key": json_key,  # 디버깅용
                })
            except Exception as e:
                print(f"Failed to load {json_key}: {e}")
                continue

        return {
            "query": {
                "image_url": image_url,
                "category": category.upper(),
                "collection_used": col_name,
            },
            "product_ids": [item["product_id"] for item in items if item["product_id"]],
        }
