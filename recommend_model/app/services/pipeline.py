# AI 추천의 핵심 계산 엔진
import torch
from PIL import Image
from io import BytesIO
from typing import Optional
import torch
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, Any

# -----------------------
# Constants
# -----------------------
REQ_TOP = "TOP"
REQ_PANTS = "PANTS"

TOP_MAIN = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "coat", "cape"
}
TOP_PART = {"sleeve", "neckline", "collar", "lapel", "epaulette", "hood", "tie"}

BOTTOM_MAIN = {"pants", "shorts", "skirt"}
BOTTOM_PART = {"pocket", "buckle"}

ONEPIECE_MAIN = {"dress", "jumpsuit"}  # PANTS 요청에 포함


# -----------------------
# Image load (S3 bytes -> PIL)
# S3에서 받은 bytes를 PIL Image로 디코딩(RGB 변환)
# -----------------------
def load_image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGB")


# -----------------------
# Crop helpers
# -----------------------
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
    """
    dets: (label_name, score, label_id, box_xyxy) list
    return: (label_name, score, label_id, box_xyxy, pad_override, flags) or None
    """
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


# -----------------------
# Detection + single crop (YOLOS)
# 탐지 결과 중 “TOP/PANTS 규칙”으로 딱 1개만 선택해서 crop
# -----------------------
@torch.no_grad()
def detect_one_crop(
    image: Image.Image,
    yolos_rt,                    # app.models_runtime.yolos.YolosRuntime
    score_thresh: float = 0.3,
    pad: int = 10,
    category: Optional[str] = None,
):
    inputs = yolos_rt.image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(yolos_rt.device) for k, v in inputs.items()}
    outputs = yolos_rt.model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=yolos_rt.device)
    results = yolos_rt.image_processor.post_process_object_detection(
        outputs, threshold=score_thresh, target_sizes=target_sizes
    )[0]

    boxes  = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()
    labels = results["labels"].detach().cpu()

    if len(scores) == 0:
        return None

    dets = []
    for i in range(len(scores)):
        label_id = int(labels[i])
        label_name = yolos_rt.model.config.id2label[label_id]
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


# -----------------------
# Embedding (FashionCLIP vision) -> 512
# crop 이미지에서 임베딩(512)을 생성
# -----------------------
@torch.no_grad()
def embed_one_crop(crop_item, fclip_rt, l2_normalize: bool = True):
    inputs = fclip_rt.clip_processor(images=crop_item["crop"], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(fclip_rt.device)

    out = fclip_rt.clip_vision(pixel_values=pixel_values)
    emb = out.image_embeds[0].detach().cpu()  # torch tensor (512,)

    if l2_normalize:
        emb = emb / (emb.norm(p=2) + 1e-12)

    return {**crop_item, "embedding": emb}


# -----------------------
# Pipeline: S3 -> crop -> embed (항상 1개 목표)
# 위 과정을 한 번에 수행하는 “오케스트레이션 함수”
# -----------------------
def run_pipeline_from_bytes(
    img_bytes: bytes,
    category: str,
    yolos_rt,
    fclip_rt,
    score_thresh: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """
    bytes -> PIL -> YOLOS crop(1개) -> FashionCLIP embedding
    (S3/Chroma 접근 없음)
    """
    image = load_image_from_bytes(img_bytes)

    crop_item = detect_one_crop(
        image=image,
        yolos_rt=yolos_rt,
        score_thresh=score_thresh,
        category=category,
    )
    if crop_item is None:
        return None

    return embed_one_crop(crop_item=crop_item, fclip_rt=fclip_rt, l2_normalize=True)
#여기서 가장 느린 부분은 모델 추론(특히 GPU 없으면 더 느림)과 S3 다운로드입니다.

#동기 처리로 운영하면 동시 요청이 많을 때 병목이 생길 수 있으니, 
# 운영에서는 workers/큐/캐싱 전략이 필요합니다(요청하신 범위에서는 구조/모듈 설명 중심으로만 정리).