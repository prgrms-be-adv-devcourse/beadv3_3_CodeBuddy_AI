# 서비스 전체에서 공통으로 쓰는 설정 값(장비, DB 경로, 컬렉션 명, AWS 인증 정보 등)을 관리하는 모듈
# 주의 사항: 
# Access Key/Secret Key를 코드에 하드코딩하지 말고 환경변수/Secret Manager/IAM Role로 주입하는 게 안전
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

# BaseModel을 상속하여 데이터 모델을 정의하고, 해당 모델의 필드와 유효성 검사 규칙을 설정할 수 있다.
class Settings(BaseSettings):
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", 8000))
    aws_access_key_id: str | None = None  # IAM Role 우선
    aws_secret_access_key: str | None = None
    aws_region: str = "ap-northeast-2"
    chroma_dir: str = "./chroma_data"
    

settings = Settings()

