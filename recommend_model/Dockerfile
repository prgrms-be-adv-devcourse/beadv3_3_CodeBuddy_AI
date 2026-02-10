FROM python:3.12-slim

# 시스템 의존성 (torch 컴파일용)
RUN apt-get update && apt-get install -y \
    gcc g++ make git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 먼저 복사 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

EXPOSE 8000

# 컨테이너 시작 시 "실행할 명령어"
# uvicorn: FastAPI 서버 실행 프로그램
# app.main:app: app/main.py 의 app 객체를 실행
# --host 0.0.0.0: 중요: 모든 IP에서 접근 허용
# --port 8000 : 컨테이너 내부에서 8000번 포트로 실행
# --reload: 개발용으로 코드 변경 시 자동 재시작
# 8001번 포트는 브라우저에서 접속하는 포트를 의미하고 8000번 포트는 fast api의 컨테이너 포트를 의미하는 것이므로 dockerfile에서는 8000 포트로 지정
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
