from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
import os
import asyncio
from .adapters.s3_adapter import S3Adapter
from .adapters.chroma_adapter import ChromaAdapter
from .services.recommend_service import RecommendService
from .api.v2.routers import api_router
from .models_runtime.yolos import YolosRuntime
from .models_runtime.fashionclip import FashionClipRuntime

from .exception.handlers import install_exception_handlers
from .kafka.kafka_producer import RecommendKafkaProducer
from .kafka.kafka_consumer import RecommendKafkaConsumer

app = FastAPI(title="Recommendation API")

install_exception_handlers(app)
app.include_router(api_router, prefix="/v2")

@asynccontextmanager
async def lifespan(app: FastAPI):

    # 1) chroma
    # docker-compose.yml의 environment에서 설정한 값을 읽어옴
    chroma = ChromaAdapter(
        host=os.getenv("CHROMA_HOST", "chromadb"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
    )
    

    # 2) S3 init
    bucket = os.environ["AWS_S3_BUCKET_NAME"]
    region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
    endpoint_url = os.getenv("S3_ENDPOINT_URL")

    s3 = S3Adapter(
        bucket_name=bucket,
        region_name=region,
        endpoint_url=endpoint_url,
    )

    # 모델 초기화
    app.state.yolos_rt = YolosRuntime(device="cpu")
    app.state.fclip_rt = FashionClipRuntime(device="cpu")
    
    # 서비스 초기화
    app.state.recommend_service = RecommendService(
        chroma=chroma,
        s3=s3,
        yolos_rt=app.state.yolos_rt,
        fclip_rt=app.state.fclip_rt
    )

    # 3) Kafka 초기화
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    producer = RecommendKafkaProducer(bootstrap_servers)
    await producer.start()

    consumer = RecommendKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        recommend_service=app.state.recommend_service,
        producer=producer,
    )
    await consumer.start()

    # 백그라운드에서 Kafka 메시지 소비 시작
    consume_task = asyncio.create_task(consumer.consume_loop())
    
    yield

    # 종료 시 Kafka 정리
    consume_task.cancel()
    await consumer.stop()
    await producer.stop()
    print("Shutting down Kafka and services...")

app.router.lifespan_context = lifespan
