#FastAPI와 Docker ChromaDB 서버 사이의 연결 어댑터

import chromadb
from ..exception.errors import ChromaQueryError

class ChromaAdapter:
    def __init__(self, host: str, port: int):
        # ChromaDB 서버와 HTTP로 대화할 클라이언트 객체
        self.client = chromadb.HttpClient(host=host, port=port)

    # 컬렉션 가져오기
    #컬렉션이 있으면 가져오고 없으면 새로 만드는 메서드. 해당 컬렉션 이름의 컬렉션을 가져온다.
    def get_collection(self, name: str):
        return self.client.get_or_create_collection(name) 

    # 벡터 검색 query(핵심 메서드)
    def query(self, collection_name: str, query_embedding: list, n_results: int, where: dict | None = None):
        try:
            col = self.get_collection(collection_name)
            return col.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            raise ChromaQueryError(collection=collection_name, reason=str(e)) from e