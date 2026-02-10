# app/adapters/s3_adapter.py
from __future__ import annotations

import json
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config


class S3Adapter:
    def __init__(
        self,
        region_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,   # 로컬 S3(minio) 쓰면 사용
        connect_timeout: int = 5,
        read_timeout: int = 30,
        max_attempts: int = 3,
    ):
        # credentials가 None이면 boto3 기본 credential chain(IAM Role, env, ~/.aws 등)을 사용합니다. [web:83]
        self._client = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            config=Config(
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retries={"max_attempts": max_attempts, "mode": "standard"},
            ),
        )

    def get_bytes(self, bucket: str, key: str) -> bytes:
        """
        S3 객체를 bytes로 반환.
        - 성공: bytes
        - 실패: 예외(원인 메시지 포함)
        """
        try:
            resp = self._client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"]  # StreamingBody
            return body.read()   # 전체를 한 번에 읽음.
        except NoCredentialsError as e:
            raise RuntimeError(
                "S3 credentials not found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or run on an environment with IAM Role."
            ) from e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            # 자주 나오는 코드: NoSuchKey, AccessDenied, InvalidAccessKeyId 등
            raise RuntimeError(f"S3 get_object failed: {code} - {msg} (bucket={bucket}, key={key})") from e

    def get_json(self, bucket: str, key: str) -> dict[str, Any]:
        """
        S3의 json 파일을 dict로 반환. -> S3에서 내려받을 때에는 바이트로 가지고오는데 이 바이트를 JSON으로 해석해서 파이썬 dict 형태로 변환을 해주는 편의 함수.
        """
        b = self.get_bytes(bucket=bucket, key=key)
        try:
            return json.loads(b)  # bytes도 json.loads가 처리 가능
        except json.JSONDecodeError as e:
            # 라벨 파일이 깨졌거나, 실제로 json이 아닌 파일을 읽은 경우
            raise RuntimeError(f"Invalid JSON in S3 object (bucket={bucket}, key={key})") from e
