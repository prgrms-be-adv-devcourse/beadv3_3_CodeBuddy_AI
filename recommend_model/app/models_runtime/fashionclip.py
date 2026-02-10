# FashionCLIP의 비전 인코더와 전처리기(CLIPProcessor)를 1회 로딩 후 재사용한다.
# 결과로 이미지 임베딩(512차원)을 뽑기 위한 엔진


from transformers import CLIPProcessor, CLIPVisionModelWithProjection

FCLIP_ID = "patrickjohncyh/fashion-clip"

class FashionClipRuntime:
    def __init__(self, device: str):
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained(FCLIP_ID)
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(FCLIP_ID).to(device)
        self.clip_vision.eval()

#주의사항
# 임베딩은 일반적으로 벡터 검색에서 정규화(L2 normalize)를 많이 합니다. 
# 정규화 여부가 달라지면 “기존에 적재한 임베딩”과 “실시간 임베딩”의 검색 결과가 달라지니, 
# 인덱싱/서빙 양쪽에서 동일하게 맞춰야 합니다.