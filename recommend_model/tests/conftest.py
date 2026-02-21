# 테스트 전반에서 공통으로 쓰는 준비 및 정리 코드를 한 곳에 모아두는 파일
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

from app.api.v2.endpoints.recommend import get_service



class DefaultFakeService:
    def recommend_many(self, reqs, topk: int = 4):
        return [{"image_url": r.image_url, "category": r.category.upper(), "product_ids": []} for r in reqs]

@pytest.fixture(autouse=True)
def _override_get_service_for_api_tests(request):
    # api 폴더 테스트에서만 자동 적용하고 싶으면 파일 경로로 분기
    if "tests\\api\\" in str(request.fspath):
        app.dependency_overrides[get_service] = lambda: DefaultFakeService()
    yield
    # _reset_dependency_overrides가 clear 해주지만, 명시해도 안전
    
'''
모든 테스트가 끝날 때 마다 app.dependency_overrides를 초기화해서, 한 테스트에서 설정한 override가 다음 테스트에 영향을 주지 않게 하는 메서드
'''
@pytest.fixture(autouse=True)
def _reset_dependency_overrides():
    yield
    app.dependency_overrides.clear()  # 테스트 후 원복


'''
테스트에서 매번 TestClient를 만들지 않고, client를 받아서 바로 API 호출을 하게 해주는 공통 fixture
'''
@pytest.fixture
def client():
    # 예외 핸들러 응답을 검증하려면 False가 유리
    return TestClient(app, raise_server_exceptions=False)
