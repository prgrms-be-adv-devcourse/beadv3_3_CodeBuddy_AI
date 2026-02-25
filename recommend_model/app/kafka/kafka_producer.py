import json
import logging
from aiokafka import AIOKafkaProducer
from .kafka_schemas import RecommendResultPayload

logger = logging.getLogger("uvicorn.error")


class RecommendKafkaProducer:

    def __init__(self, bootstrap_servers: str):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )

    async def start(self):
        await self.producer.start()
        logger.info("Kafka Producer 시작")

    async def stop(self):
        await self.producer.stop()
        logger.info("Kafka Producer 종료")

    async def send_result(self, result: RecommendResultPayload):
        payload = result.model_dump()
        await self.producer.send("recommend.result", value=payload)
        logger.info(f"추천 결과 발행: requestId={result.requestId}")
