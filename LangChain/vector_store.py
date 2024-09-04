import glob
import os

import pandas as pd
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


class MyFAISS():
    def __init__(self):
        super().__init__()
        db = pd.read_csv('./dataset/original.csv')
        self.to_vecotr_list = db[db['to_vector'] == False]['paper_name'].tolist()

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 사용할 임베딩 함수
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )

        self.persist_directory = './dataset/vector_db'
        os.makedirs(self.persist_directory, exist_ok=True)

    def update_papers(self):
        for paper in self.to_vecotr_list:
            docs = self.get_docs(paper)
            self.create_vector_store(docs, paper)

    def get_docs(self, paper):
        loader = PyMuPDFLoader(paper)
        pages = loader.load()
        docs = self.text_splitter.split_documents(pages)

        return docs

    def create_vector_store(self, docs, store_name):
        persistent_directory = os.path.join(self.persist_directory, store_name)
        if not os.path.exists(persistent_directory):
            print(f"\n--- 벡터 저장소 {store_name} 생성 중 ---")
            FAISS.from_documents(
                docs, self.embeddings, persist_directory=persistent_directory)
            print(f"--- 벡터 저장소 {store_name} 생성 완료 ---")
        else:
            print(
                f"벡터 저장소 {store_name}이(가) 이미 존재합니다. 초기화할 필요가 없습니다.")



