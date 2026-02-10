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


def infer_token_and_folder(image_id: str, category: str) -> str | None:
    """추론 로직: image_id에서 token 찾고, category에 맞는 folder 반환"""
    image_lower = image_id.lower()
    cat = (category or "").upper()
    
    for token, cat_folders in TOKEN_TO_FOLDER.items():
        if token in image_lower:
            if cat_folders.get(cat):
                return f"label_data/{cat}/{cat_folders[cat]}/{image_id}.json"
    
    return None  # 매치 안되면 None


def backfill_label_json_key(
    client: chromadb.ClientAPI,
    collection_name: str,
    batch_size: int = 256,
    json_key_field: str = "label_json_key",
):
    col = client.get_collection(collection_name)
    total = col.count()
    print(f"Processing {total} items in {collection_name}")

    updated_count = 0
    for offset in range(0, total, batch_size):
        batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break

        new_metas = []
        for rid, meta in zip(ids, metas):
            meta = dict(meta or {})

            # DB ids == image_id 형태라 통일
            image_id = meta.get("image_id") or rid
            category = meta.get("category")  # TOP / PANTS

            # ✅ 고친 부분: 올바른 순서로 호출 (image_id, category)
            json_key = infer_token_and_folder(image_id, category)
            if json_key:
                meta[json_key_field] = json_key
                updated_count += 1
            else:
                print(f"No match: image_id={image_id}, category={category}")

            new_metas.append(meta)

        col.update(ids=ids, metadatas=new_metas)

    print(f"Updated {updated_count}/{total} items in {collection_name}")
    print("Sample metadatas:", col.get(limit=3, include=["metadatas"]))


def main():
    client = chromadb.HttpClient(host="localhost", port=8000)  # 환경에 맞게 수정
    backfill_label_json_key(client, "fashion_items")
    backfill_label_json_key(client, "fashion_items_pants")


if __name__ == "__main__":
    main()
