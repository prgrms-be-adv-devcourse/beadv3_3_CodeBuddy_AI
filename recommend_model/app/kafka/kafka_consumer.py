import json
import logging
import asyncio
from aiokafka import AIOKafkaConsumer
from .kafka_schemas import (
    RecommendRequestPayload,
    RecommendResultPayload,
    RecommendResultItemPayload,
)
from .kafka_producer import RecommendKafkaProducer
from ..services.recommend_service import RecommendService
from ..schemas.recschema import RecommendRequest

logger = logging.getLogger("uvicorn.error")


class RecommendKafkaConsumer:

    def __init__(
        self,
        bootstrap_servers: str,
        recommend_service: RecommendService,
        producer: RecommendKafkaProducer,
    ):
        self.consumer = AIOKafkaConsumer(
            "order.recommend.request",
            bootstrap_servers=bootstrap_servers,
            group_id="ai-service-group",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.service = recommend_service
        self.producer = producer

    async def start(self):
        await self.consumer.start()
        logger.info("Kafka Consumer 시작 - 'order.recommend.request' 구독 중")

    async def stop(self):
        await self.consumer.stop()
        logger.info("Kafka Consumer 종료")

    async def consume_loop(self):
        try:
            async for msg in self.consumer:
                await self._handle_message(msg.value)
        except asyncio.CancelledError:
            logger.info("Consumer 루프 종료")
        except Exception as e:
            logger.error(f"Consumer 루프 에러: {e}")

    async def _handle_message(self, raw: dict):
        try:
            # 1) Kafka JSON -> Pydantic 객체
            request = RecommendRequestPayload(**raw)
            logger.info(f"추천 요청 수신: requestId={request.requestId}")

            # 2) 기존 RecommendService가 받는 형식으로 변환
            service_reqs = [
                RecommendRequest(image_url=item.imageUrl, category=item.category)
                for item in request.items
            ]

            # 3) 기존 추천 로직 호출 (S3 -> YOLOS → FashionCLIP -> ChromaDB)
            service_results = self.service.recommend_many(service_reqs)

            # 4) 결과 -> Kafka 응답으로 변환(java 서버가 이해할 수 있게 camelcase PayLoad로 응답)
            result_items = [
                RecommendResultItemPayload(
                    imageUrl=r.image_url,
                    category=r.category,
                    productIds=r.product_ids,
                )
                for r in service_results
            ]

            # 5) 성공 결과 발행
            result = RecommendResultPayload(
                requestId=request.requestId,
                memberId=request.memberId,
                success=True,
                failReason=None,
                result=result_items,
            )
            await self.producer.send_result(result)

        except Exception as e:
            # 6) 실패 결과 발행
            logger.error(f"추천 처리 실패: {e}")
            error_result = RecommendResultPayload(
                requestId=raw.get("requestId", "unknown"),
                memberId=raw.get("memberId", 0),
                success=False,
                failReason=str(e),
                result=None,
            )
            await self.producer.send_result(error_result)
