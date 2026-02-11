from __future__ import annotations
import json, os
from typing import Any, Optional, List
from pathlib import Path
import logging

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

logger = logging.getLogger(__name__)

class S3Adapter:
    def __init__(self, bucket_name: str, region_name: str = "ap-northeast-2", endpoint_url: Optional[str] = None):

        # AWS 자격 증명(환경변수)
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        self.bucket_name = bucket_name

        #S3 서비스와 통신할 수 있는 클라이언트 객체 생성.
        # 설정 저장하고 
        self._client = boto3.client(
            "s3", region_name=region_name, aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key, endpoint_url=endpoint_url,
            config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 3}),
        )
        self._client.list_buckets()  # 연결 테스트

    #S3 객체를 바이트로 다운로드
    def get_bytes(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket_name, Key=key)
        return resp["Body"].read()

    # S3 JSON 파일을 Python dict로 파싱
    def get_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(key))

    #S3 Presigned URL 생성 (클라이언트에서 이미지 직접 접근을 위해서)
    def get_image_url(self, key: str, expires_in: int = 3600) -> str:
        return self._client.generate_presigned_url(
            'get_object', Params={'Bucket': self.bucket_name, 'Key': key}, ExpiresIn=expires_in
        )

    # S3에서 실시간으로 JSON 파일 목록을 조회하는 함수
    def list_category_jsons(self, category: str) -> List[str]:
        prefix = f"label_data/{category.upper()}/"
        paginator = self._client.get_paginator('list_objects_v2')
        json_keys = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            if 'Contents' in page:
                json_keys.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.json')])
        return json_keys

    # Chroma 메타의 label_json_key로 JSON 로드
    def get_product_info(self, json_key: str) -> dict[str, Any]:
        
        try:
            return self.get_json(json_key)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchKey":
                raise RuntimeError(f"S3 key not found: {json_key}")
            raise

    # JSON과 이미지 간 연관관계 탐색/검증용 -> 데이터 관리 측면에서 사용하는 함수
    # JSON 파일 안의 이미지 경로가 S3에 실제로 존재하는지 확인하기 위해서 경로만 뽑아서 S3 key 형태로 정규화해 반환하는 함수
    def get_image_key_from_json(self, json_key: str) -> str:

        data = self.get_json(json_key) # JSON 로드
        img_key = data.get('image_url') or data.get('image_path') or data.get('front_image') # 키 탐색

        # s3:// 형식이면 S3 key만 추출
        if not img_key:
            raise ValueError(f"No image key in {json_key}: {data}")
        
        if not img_key.startswith('s3://'):
            img_key = f"s3://{self.bucket_name}/{img_key.lstrip('/')}"
        return img_key.replace('s3://', '').split('/', 1)[1]  # 버킷명 제거


    # 디버깅용 JSON 목록
    # S3에서 해당 json 파일이 없는 경우, 에러 메시지와 함께 같은 카테고리 아래 실제 존재하는 JSON 파일 3개의 경로를 추출해주는 함수
    # 경로가 잘못된 경우
    def _list_prefix_jsons(self, prefix: str, limit: int = 3) -> list[str]:
        """디버깅용 JSON 목록"""
        paginator = self._client.get_paginator("list_objects_v2")
        jsons = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    jsons.append(obj["Key"])
                    if len(jsons) >= limit:
                        return jsons
        return jsons