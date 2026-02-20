'''
FastAPI 라우터나 실제 S3/Chroma/모델을 붙이지 않고, 서비스 로직이 입력에 따라 어떤 예외를 던지는지, 내부 예외를 우리 도메인 예외로 잘 감싸는지를 빠르고 안정적으로 검증하는 테스트들
'''

import pytest
from unittest.mock import Mock

from app.services.recommend_service import RecommendService
from app.exception.errors import UnsupportedCategory, PipelineError, ChromaQueryError

'''
카테고리가 "TOP"/"PANTS"가 아닌 값으로 들어오면 RecommendService.recommend()가 UnsupportedCategory 예외를 던지는지 확인하는 메서드
'''
def test_recommend_unsupported_category():
    svc = RecommendService(chroma=Mock(), s3=Mock(), yolos_rt=Mock(), fclip_rt=Mock())
    with pytest.raises(UnsupportedCategory):
        svc.recommend("s3://bucket/a.jpg", "HAT")

'''
파이프라인 함수에서 예상치 못한 예외가 나도 서비스는 이를 그대로 흘리지 않고 PipelineError로 감싸서 던지는지 확인
'''
def test_recommend_pipeline_error_wrapped(monkeypatch):
    mock_s3 = Mock()
    mock_s3.get_bytes.return_value = b"fake" # 실제 S3에서 다운로드하지 않고도 서비스가 다음 단계(파이프라인)로 진행하도록 이미지 bytes를 가짜로 공급

    svc = RecommendService(chroma=Mock(), s3=mock_s3, yolos_rt=Mock(), fclip_rt=Mock())

    def boom(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr("app.services.recommend_service.run_pipeline_from_bytes", boom)

    with pytest.raises(PipelineError):
        svc.recommend("s3://bucket/a.jpg", "TOP")

'''
파이프라인은 성공해서 임베딩이 나왔는데, Chroma 쿼리에서 예외가 터지면 서비스가 이를 ChromaQueryError로 감싸서 던지는지 확인하는 메서드.
'''
def test_recommend_chroma_error_wrapped(monkeypatch):
    mock_s3 = Mock()
    mock_s3.get_bytes.return_value = b"fake"

    class FakeEmbedding:
        def numpy(self): return self
        def tolist(self): return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "app.services.recommend_service.run_pipeline_from_bytes",
        lambda **kwargs: {"embedding": FakeEmbedding()},
    )

    mock_chroma = Mock()
    mock_chroma.query.side_effect = Exception("chroma down")

    svc = RecommendService(chroma=mock_chroma, s3=mock_s3, yolos_rt=Mock(), fclip_rt=Mock())

    with pytest.raises(ChromaQueryError):
        svc.recommend("s3://bucket/a.jpg", "TOP")
