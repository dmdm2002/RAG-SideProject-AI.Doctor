import os
import pandas as pd

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


class VectorStore:
    def __init__(self):
        super().__init__()
        self.enroll_db = pd.read_csv('LangChain/dataset/enroll_db.csv')
        self.to_vecotr_list = self.enroll_db[self.enroll_db['to_vector'] == False]['paper_name'].tolist()

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 사용할 임베딩 함수
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        self.persist_directory = 'LangChain/dataset/vector_db'
        os.makedirs(self.persist_directory, exist_ok=True)

    def update_papers(self):
        for paper in self.to_vecotr_list:
            docs = self.get_docs(f'LangChain/dataset/raw_data/{paper}')
            self.create_vector_store(docs, paper)

    def get_docs(self, paper_path):
        loader = PyMuPDFLoader(paper_path)
        pages = loader.load()
        docs = self.text_splitter.split_documents(pages)
        print(docs)

        return docs

    def create_vector_store(self, docs, store_name):
        if not docs:
            print(f"No documents to create vector store for {store_name}")
            return

        persistent_directory = os.path.join(self.persist_directory, store_name)
        if not os.path.exists(persistent_directory):
            print(f"\n--- 벡터 저장소 {store_name} 생성 중 ---")
            Chroma.from_documents(
                docs, self.embeddings, persist_directory=persistent_directory)
            print(f"--- 벡터 저장소 {store_name} 생성 완료 ---")
            # self.enroll_db.loc[self.enroll_db['paper_name'] == store_name, 'to_vector'] = True
        else:
            print(
                f"벡터 저장소 {store_name}이(가) 이미 존재합니다. 초기화할 필요가 없습니다.")

    def query_vector_store(self):
        if os.path.exists(self.persist_directory):
            print(f"---- 벡터 저장소에 쿼리 실행중 ----")
            db = Chroma(persist_directory=self.persist_directory,
                        embedding_function=self.embeddings)

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            )

            return retriever

        else:
            print(f"벡터 저장소에 데이터가 존재하지 않습니다.")




