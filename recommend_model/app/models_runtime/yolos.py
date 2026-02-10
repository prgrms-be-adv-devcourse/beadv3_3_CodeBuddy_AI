# YOLOS 탐지 모델과 전처리기(AutoImageProcessor)를 한 번만 로딩해서 재사용하도록 보관한다.
from transformers import AutoImageProcessor, AutoModelForObjectDetection

YOLOS_ID = "valentinafeve/yolos-fashionpedia"

class YolosRuntime:
    def __init__(self, device: str):
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(YOLOS_ID) # 이미지 -> 텐서 변환 및 post-process 기능 제공
        self.model = AutoModelForObjectDetection.from_pretrained(YOLOS_ID).to(device) # 탐지 모델(CPU에 올라간 상태)
        self.model.eval() # 추론 모드 고정
