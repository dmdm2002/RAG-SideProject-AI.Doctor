import openai
import os
from langchain_core.messages import HumanMessage, SystemMessage

from LangChain.vector_store import VectorStore
from LangChain.database_handler import DatabaseHandler
# from LangChain.rag_chain_builder import RAGChinBuilder
from LangChain.rag_multimodal_chain_builder import RAGChinBuilder


def continual_chat():
    rag_chain_builder = RAGChinBuilder()
    rag_chain, image_caption = rag_chain_builder.get_rag_chain(image_path='./Vision/ct_images/images/labeled/4_jpg.rf.e6fdccf8963d26789efd99709b1e5f43.jpg',
                                                               prompt_text=None)

    print("AI와 대화를 시작하세요! 대화를 종료하려면 'exit'를 입력하세요.")
    chat_history = []  # 대화 기록을 수집합니다 (메시지의 시퀀스)
    while True:
        query = input("You: ")

        print("CT 사진 소견 분석 결과와 질문 합성...")
        query = f'{query}, 해당 환자의 CT사진 촬영 결과: {image_caption}'

        print(f'최종 질문: {query}')
        if query.lower() == "exit":
            break

        # 사용자의 질문을 검색 체인으로 처리
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # AI의 응답을 출력합니다
        print(f"AI: {result['answer']}")
        # U대화 기록을 업데이트합니다
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


os.environ['OPENAI_API_KEY'] = ""
api_key = os.getenv('OPENAI_API_KEY')

openai.api_key = api_key

db_hanlder = DatabaseHandler()
db_hanlder.update_db()

my_chroma = VectorStore()
my_chroma.update_papers()

continual_chat()

