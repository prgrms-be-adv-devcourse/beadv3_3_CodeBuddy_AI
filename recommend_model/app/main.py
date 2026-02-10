# 역할
# FastAPI 앱을 만들고 라우터를 등록합니다.

# startup에서 S3/Chroma/모델/서비스 객체를 생성해 app.state에 저장합니다.

from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager

from core.config import settings
from adapters.s3_adapter import S3Adapter
from adapters.chroma_adapter import ChromaAdapter
from services.recommend_service import RecommendService  # 추가!
from api.v2.routers import api_router
from models_runtime.yolos import YolosRuntime  # 실제 클래스명 확인
from models_runtime.fashionclip import FashionClipRuntime

app = FastAPI(title="Recommendation API") #메인 FastAPI 앱
app.include_router(api_router, prefix="/v2") #모든 경로에서 포트 다음에 v2를 고정적으로 넣고 그 다음에 fast api의 라우터가 지정한 경로값이 들어간다.

# lifespan()의 startup 코드 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    app.state.s3 = S3Adapter(
        aws_access_key_id=settings.aws_access_key_id,  # None이면 IAM Role
        aws_secret_access_key=settings.aws_secret_access_key, # 이미지 로드
        region_name=settings.aws_region, # YOLO+CLIP
    )
    app.state.chroma = ChromaAdapter(
        host=settings.chroma_host, # "chromadb"
        port=settings.chroma_port, # 8000
    )
    # 🎯 모델 초기화 (여기서 1회 로딩)
    app.state.yolos_rt = YolosRuntime(device="cpu")
    app.state.fclip_rt = FashionClipRuntime(device="cpu")
    
    
    # 서비스 초기화 (app.state 사용)
    app.state.recommend_service = RecommendService(
        chroma=app.state.chroma,
        s3_adapter=app.state.s3,  # S3Adapter 전달
        yolos_rt=app.state.yolos_rt,
        fclip_rt=app.state.fclip_rt
    )
    yield
    print("Shutting down models and services...")

app.router.lifespan_context = lifespan  # FastAPI 0.95+ 방식