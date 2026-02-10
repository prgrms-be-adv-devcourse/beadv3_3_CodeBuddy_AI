#FastAPI와 Docker ChromaDB 서버 사이의 **연결 어댑터

#서비스 로직은 “검색한다/넣는다”만 알고, Chroma의 구체 API 호출은 몰라도 되게 합니다.

import chromadb

class ChromaAdapter:
    def __init__(self, host: str, port: int):
        # ChromaDB 서버와 HTTP로 대화할 클라이언트 객체
        self.client = chromadb.HttpClient(host=host, port=port)

    #컬렉션 가져오기
    def get_collection(self, name: str):
        return self.client.get_or_create_collection(name) #컬렉션이 있으면 가져오고 없으면 새로 만드는 메서드. 해당 컬렉션 이름의 컬렉션을 가져온다.

    # 벡터 검색 query(핵심 메서드)
    def query(self, collection_name: str, query_embedding: list, n_results: int, where: dict | None = None):
        col = self.get_collection(collection_name)
        return col.query(
            query_embeddings=[query_embedding], # 검색할 임베딩 (512차원)
            n_results=n_results, # 상위 N개 결과 ex) n_results: 4  # 상위 4개 추천
            where=where, # 메타데이터 필터 ex) where: {"category": "TOP"}  # TOP 카테고리만 검색
        )


#주의사항
# “추천 결과에서 이미지 다시 찾기”를 쉽게 하려면, 인덱싱 시 metadatas에 최소 bucket, key, category를 넣어두는 게 좋습니다.

# distance가 “작을수록 유사”인지 “클수록 유사”인지는 사용 설정/거리 함수에 따라 다를 수 있어, 
# 응답 필드명을 score 대신 distance로 두는 게 안전합니다(저도 그렇게 구성했습니다).