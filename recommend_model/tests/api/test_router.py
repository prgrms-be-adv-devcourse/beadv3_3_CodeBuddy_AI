'''
API(라우터) 테스트 파일
/v2/recommend 엔드포인트의 요청/응답 계약 + 예외 응답 핸들링을 검증하는 API 테스트 파일

FastAPI의 dependency_overrides: Depends(get_service)가 실제 RecommendService를 호출하지 않고, 테스트용 FakeService로 교체된다.
이를 통해 S3/Chroma/모델 같은 외부 의존성 없이 순수 라우터 동작만 검증할 수 있다.
'''
from app.main import app
from app.api.v2.endpoints.recommend import get_service
from app.exception.errors import UnsupportedCategory, S3KeyNotFound, PipelineError, ChromaQueryError

class FakeOKService:
    def recommend_many(self, reqs, topk: int = 4):
        return [
            {"image_url": r.image_url, "category": r.category.upper(), "product_ids": ["p1", "p2"]}
            for r in reqs
        ]

class FakeUnsupportedService:
    def recommend_many(self, reqs, topk: int = 4):
        raise UnsupportedCategory(reqs[0].category)

class FakeS3NotFoundService:
    def recommend_many(self, reqs, topk: int = 4):
        raise S3KeyNotFound("label_data/TOP/missing.jpg")

class FakePipelineErrorService:
    def recommend_many(self, reqs, topk: int = 4):
        raise PipelineError("No detection found")

class FakeChromaErrorService:
    def recommend_many(self, reqs, topk: int = 4):
        raise ChromaQueryError(collection="fashion_items", reason="timeout")

def test_post_recommend_ok(client):
    app.dependency_overrides[get_service] = lambda: FakeOKService()  # override [web:1]

    payload = [{"image_url": "s3://bucket/a.jpg", "category": "TOP"}]
    r = client.post("/v2/recommend", json=payload)

    assert r.status_code == 200
    assert r.json() == [{"image_url": "s3://bucket/a.jpg", "category": "TOP", "product_ids": ["p1", "p2"]}]

def test_post_recommend_validation_error_custom_format(client):
    # RequestValidationError -> handler가 422 + error wrapper로 변환
    payload = [{"image_url": "s3://bucket/a.jpg"}]
    r = client.post("/v2/recommend", json=payload)

    print("status:", r.status_code)
    print("body:", r.text)   # 핵심

    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["message"] == "Invalid request."
    assert "errors" in body["error"]["detail"]

def test_post_recommend_unsupported_category(client):
    app.dependency_overrides[get_service] = lambda: FakeUnsupportedService()

    payload = [{"image_url": "s3://bucket/a.jpg", "category": "HAT"}]
    r = client.post("/v2/recommend", json=payload)

    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "UNSUPPORTED_CATEGORY"
    assert body["error"]["message"] == "Unsupported category."
    assert body["error"]["detail"] == {"category": "HAT"}

def test_post_recommend_s3_key_not_found(client):
    app.dependency_overrides[get_service] = lambda: FakeS3NotFoundService()

    payload = [{"image_url": "s3://bucket/missing.jpg", "category": "TOP"}]
    r = client.post("/v2/recommend", json=payload)

    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "S3_KEY_NOT_FOUND"
    assert body["error"]["message"] == "S3 object not found."
    assert body["error"]["detail"] == {"key": "label_data/TOP/missing.jpg"}

def test_post_recommend_pipeline_error(client):
    app.dependency_overrides[get_service] = lambda: FakePipelineErrorService()

    payload = [{"image_url": "s3://bucket/a.jpg", "category": "TOP"}]
    r = client.post("/v2/recommend", json=payload)

    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "PIPELINE_ERROR"
    assert body["error"]["message"] == "Failed to run image pipeline."
    assert body["error"]["detail"] == {"reason": "No detection found"}

def test_post_recommend_chroma_query_error(client):
    app.dependency_overrides[get_service] = lambda: FakeChromaErrorService()

    payload = [{"image_url": "s3://bucket/a.jpg", "category": "TOP"}]
    r = client.post("/v2/recommend", json=payload)

    assert r.status_code == 502
    body = r.json()
    assert body["error"]["code"] == "CHROMA_QUERY_ERROR"
    assert body["error"]["message"] == "Vector DB query failed."
    assert body["error"]["detail"] == {"collection": "fashion_items", "reason": "timeout"}
