import chromadb
import re


# category + token -> label folder
FOLDER_BY_TOKEN = {
    "TOP": {
        "sweater": "VL_TOP_sweater",
        "coat": "VL_TOP_coat",
    },
    "PANTS": {
        "bottom": "VL_PANTS_bottom",
        "onepiece_dress": "VL_PANTS_onepiece_dress",
        "jumpsuite": "VL_PANTS_jumpsuite",
    },
}


# TOP/PANTS별로 "폴더 결정을 위해 허용하는 토큰"만 정의
ALLOWED_TOKENS_BY_CATEGORY = {
    "TOP": ["sweater", "coat"],
    "PANTS": ["bottom", "onepiece_dress", "jumpsuite"],
}


TOKEN_TO_FOLDER = {
    "sweater": {"TOP": "VL_TOP_sweater"},
    "coat": {"TOP": "VL_TOP_coat"},
    "bottom": {"PANTS": "VL_PANTS_bottom"},
    "onepiece": {"PANTS": "VL_PANTS_onepiece_dress"},  # (dress) 변형도 onepiece로 통합
    "dress": {"PANTS": "VL_PANTS_onepiece_dress"},
    "jumpsuite": {"PANTS": "VL_PANTS_jumpsuite"},
}

import chromadb

def _infer_label_folder_from_image_id(image_id: str, category: str) -> str | None:
    img = image_id.lower()
    cat = category.upper()

    if cat == "TOP":
        if "coat" in img:
            return "VL_TOP_coat"
        if "sweater" in img:
            return "VL_TOP_sweater"
        return None

    if cat == "PANTS":
        if "onepiece" in img:
            if "jumpsuite" in img:
                return "VL_PANTS_onepiece_jumpsuite"
            if "dress" in img or "(dress)" in img:
                return "VL_PANTS_onepiece_dress"
            return "VL_PANTS_onepiece_dress"  # 기본값
        if "bottom" in img or "pants" in img:
            return "VL_PANTS_bottom"
        return None

    return None

def infer_label_json_key(image_id: str, category: str) -> str:
    folder = _infer_label_folder_from_image_id(image_id, category)
    if not folder:
        raise ValueError(f"Cannot infer folder for image_id={image_id}, category={category}")
    return f"label_data/{category.upper()}/{folder}/{image_id}.json"

def backfill_label_json_key(client: chromadb.ClientAPI, collection_name: str, batch_size: int = 256):
    col = client.get_collection(collection_name)
    total = col.count()
    print(f" Backfilling {collection_name}: {total} items")

    updated = 0
    for offset in range(0, total, batch_size):
        batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        
        new_metas = []
        for i, meta in enumerate(metas):
            meta = dict(meta)
            image_id = meta.get("image_id") or ids[i]
            category = meta.get("category")
            
            try:
                json_key = infer_label_json_key(image_id, category)
                meta["label_json_key"] = json_key
                updated += 1
            except ValueError as e:
                print(f" SKIP: {e}")
            
            new_metas.append(meta)
        
        col.update(ids=ids, metadatas=new_metas)  # 전체 덮어쓰기
    
    print(f" UPDATED: {updated}/{total} in {collection_name}")

def verify(client: chromadb.ClientAPI):
    """덮어쓰기 확인"""
    for coll in ["fashion_items", "fashion_items_pants"]:
        col = client.get_collection(coll)
        sample = col.get(limit=3, include=["metadatas"])
        print(f"\n {coll} sample:")
        for meta in sample.get("metadatas", []):
            print(f"  label_json_key: {meta.get('label_json_key')}")

def main():
    client = chromadb.HttpClient(host="localhost", port=8000)
    
    print(" Starting backfill...")
    backfill_label_json_key(client, "fashion_items")
    backfill_label_json_key(client, "fashion_items_pants")
    
    print("\n VERIFICATION:")
    verify(client)

if __name__ == "__main__":
    main()