from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager

from app.core.config import settings
from .adapters.chroma_adapter import ChromaAdapter
from .services.recommend_service import RecommendService
from .api.v2.routers import api_router
from .models_runtime.yolos import YolosRuntime
from .models_runtime.fashionclip import FashionClipRuntime

app = FastAPI(title="Recommendation API")
app.include_router(api_router, prefix="/v2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup - S3 제거, 로컬 파일만
    app.state.chroma = ChromaAdapter(
        host=settings.chroma_host,  # "chromadb"
        port=settings.chroma_port,  # 8000
    )
    
    # 모델 초기화 (변경 없음)
    app.state.yolos_rt = YolosRuntime(device="cpu")
    app.state.fclip_rt = FashionClipRuntime(device="cpu")
    
    # 서비스 초기화 (S3 없이 로컬 파일만 처리하도록 수정 필요)
    app.state.recommend_service = RecommendService(
        chroma=app.state.chroma,
        # s3_adapter 제거 (RecommendService에서 로컬 파일 처리)
        yolos_rt=app.state.yolos_rt,
        fclip_rt=app.state.fclip_rt
    )
    
    yield
    print("Shutting down models and services...")

app.router.lifespan_context = lifespan
