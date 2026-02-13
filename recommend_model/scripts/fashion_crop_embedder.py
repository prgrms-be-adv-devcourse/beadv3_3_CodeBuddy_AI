# scripts/build_index.py
import os, glob
import torch
from PIL import Image
import chromadb
from chromadb.config import Settings
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import chromadb

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

YOLOS_ID = "valentinafeve/yolos-fashionpedia"
FCLIP_ID = "patrickjohncyh/fashion-clip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQ_TOP = "TOP"
REQ_PANTS = "PANTS"

TOP_MAIN = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "coat", "cape"
}
TOP_PART = {"sleeve", "neckline", "collar", "lapel", "epaulette", "hood", "tie"}
BOTTOM_MAIN = {"pants", "shorts", "skirt"}
BOTTOM_PART = {"pocket", "buckle"}
ONEPIECE_MAIN = {"dress", "jumpsuit"}

BATCH_SIZE = 1000  # 미니배치 크기

def union_boxes_xyxy(boxes):
    xs1 = [b[0] for b in boxes]; ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]; ys2 = [b[3] for b in boxes]
    return (min(xs1), min(ys1), max(xs2), max(ys2))

def crop_pack(image, label_name, label_id, score, box_xyxy, pad):
    W, H = image.size
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(W, int(x2) + pad)
    y2 = min(H, int(y2) + pad)
    return {
        "crop": image.crop((x1, y1, x2, y2)),
        "score": float(score),
        "label_id": int(label_id),
        "label_name": label_name,
        "box_xyxy": (x1, y1, x2, y2),
    }

def pick_one_detection(dets, category):
    if not dets:
        return None

    if category == REQ_TOP:
        mains = [d for d in dets if d[0] in TOP_MAIN]
        if mains:
            ln, sc, lid, bx = max(mains, key=lambda x: x[1])
            return (ln, sc, lid, bx, None, {"used_part_fallback": False, "is_onepiece": False})
        
        parts = [d for d in dets if d[0] in TOP_PART]
        if parts:
            u = union_boxes_xyxy([d[3] for d in parts])
            best_sc = max(d[1] for d in parts)
            return ("TOP_PART_union", best_sc, -1, u, 140, {"used_part_fallback": True, "is_onepiece": False})
        return None

    if category == REQ_PANTS:
        mains = [d for d in dets if (d[0] in BOTTOM_MAIN) or (d[0] in ONEPIECE_MAIN)]
        if mains:
            ln, sc, lid, bx = max(mains, key=lambda x: x[1])
            return (ln, sc, lid, bx, None, {"used_part_fallback": False, "is_onepiece": (ln in ONEPIECE_MAIN)})
        parts = [d for d in dets if d[0] in BOTTOM_PART]
        if parts:
            u = union_boxes_xyxy([d[3] for d in parts])
            best_sc = max(d[1] for d in parts)
            return ("BOTTOM_PART_union", best_sc, -1, u, 120, {"used_part_fallback": True, "is_onepiece": False})
        return None

    ln, sc, lid, bx = max(dets, key=lambda x: x[1])
    return (ln, sc, lid, bx, None, {"used_part_fallback": False, "is_onepiece": False})

@torch.no_grad()
def detect_one_crop(image, image_processor, det_model, score_thresh=0.3, pad=10, category=None):
    try:
        det_model.eval()
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = det_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
        results = image_processor.post_process_object_detection(
            outputs, threshold=score_thresh, target_sizes=target_sizes
        )[0]

        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        labels = results["labels"].detach().cpu()

        if len(scores) == 0:
            return None

        dets = []
        for i in range(len(scores)):
            label_id = int(labels[i])
            label_name = det_model.config.id2label[label_id]
            box_xyxy = tuple(map(int, boxes[i].tolist()))
            dets.append((label_name, float(scores[i]), label_id, box_xyxy))

        picked = pick_one_detection(dets, category)
        if picked is None:
            return None

        ln, sc, lid, bx, pad_override, flags = picked
        use_pad = pad_override if pad_override is not None else pad
        item = crop_pack(image, ln, lid, sc, bx, use_pad)
        item.update(flags)
        return item
    except Exception as e:
        logger.warning(f"Detection failed for image: {str(e)}")
        return None

@torch.no_grad()
def embed_one_crop(crop_item, clip_processor, clip_vision, l2_normalize=True):
    try:
        clip_vision.eval()
        inputs = clip_processor(images=crop_item["crop"], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        out = clip_vision(pixel_values=pixel_values)
        emb = out.image_embeds[0].detach().cpu()
        if l2_normalize:
            emb = emb / (emb.norm(p=2) + 1e-12)
        return {**crop_item, "embedding": emb}
    except Exception as e:
        logger.warning(f"Embedding failed for crop: {str(e)}")
        return None

def process_image_batch(
    paths: List[str], 
    image_processor, 
    det_model, 
    clip_processor, 
    clip_vision,
    category: str
) -> List[Dict[str, Any]]:
    """이미지 배치를 처리하여 ids, embs, metas 리스트 반환"""
    ids, embs, metas = [], [], []
    
    for p in paths:
        try:
            image_id = os.path.splitext(os.path.basename(p))[0]
            image = Image.open(p).convert("RGB")
            
            crop_item = detect_one_crop(
                image, image_processor, det_model,
                score_thresh=0.3, category=category
            )
            if crop_item is None:
                logger.debug(f"No detection for {image_id}")
                continue

            out = embed_one_crop(crop_item, clip_processor, clip_vision)
            if out is None:
                logger.debug(f"Embedding failed for {image_id}")
                continue

            emb = out["embedding"].numpy().tolist()
            ids.append(image_id)
            embs.append(emb)
            metas.append({
                "category": category,
                "image_id": image_id,
                "local_image_path": p,
                "label_name": out["label_name"],
                "used_part_fallback": out["used_part_fallback"],
                "is_onepiece": out["is_onepiece"],
            })
            
        except Exception as e:
            logger.warning(f"Failed to process {p}: {str(e)}")
            continue
    
    return ids, embs, metas

def main(
    image_glob: str,
    category: str,
    chroma_host: str = "localhost",
    chroma_port: str = "8000",
    collection_name: str = "fashion_items_pants",
    batch_size: int = BATCH_SIZE,
    chroma_dir: str = "./chroma_data", 
):
    logger.info(f"Starting index build for category: {category}")
    logger.info(f"Image glob: {image_glob}")
    
    # 모델 로드
    logger.info("Loading models...")
    image_processor = AutoImageProcessor.from_pretrained(YOLOS_ID)
    det_model = AutoModelForObjectDetection.from_pretrained(YOLOS_ID).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(FCLIP_ID)
    clip_vision = CLIPVisionModelWithProjection.from_pretrained(FCLIP_ID).to(DEVICE)
    
    
    # ChromaDB 연결
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    
    logger.info("Connecting to ChromaDB...")
    client = chromadb.HttpClient(host="localhost", port=8000)

    col = client.create_collection(name=collection_name)
    
    # 이미지 경로 수집
    paths = sorted([
        str(p) for p in Path(image_glob).rglob("*") 
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    total_images = len(paths)
    logger.info(f"Found {total_images} images")
    
    successful_upserts = 0
    total_processed = 0
    
    # 미니배치 처리
    for i in range(0, total_images, batch_size):
        batch_paths = paths[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1} "
                   f"({len(batch_paths)} images, {i+1}-{min(i+batch_size, total_images)})")
        
        try:
            ids, embs, metas = process_image_batch(
                batch_paths, image_processor, det_model, 
                clip_processor, clip_vision, category
            )
            
            if ids:
                col.upsert(ids=ids, embeddings=embs, metadatas=metas)
                successful_upserts += 1
                total_processed += len(ids)
                logger.info(f"Upserted {len(ids)} items. Total: {total_processed}")
            else:
                logger.info("No valid items in this batch")
                
        except Exception as e:
            logger.error(f"Batch {i//batch_size + 1} failed: {str(e)}")
            continue
    
    logger.info(f"Index build completed!")
    logger.info(f"Total batches processed: {successful_upserts}")
    logger.info(f"Total embeddings stored: {total_processed}")
    logger.info(f"Collection stats: {col.count()} items")

if __name__ == "__main__":
    main(
        image_glob=r"C:\Users\ehjun\Documents\데브코스데이터\temp\008.의류_통합_데이터(착용_이미지,_치수_및_원단_정보)\01-1.정식개방데이터\Validation\01.원천데이터\이미지데이터\하의",
        category="PANTS",
        chroma_host="localhost",
        chroma_port="8000", 
        batch_size=1000,
    )
