from PIL import Image
import requests
from io import BytesIO

from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from LangChain.vector_store import VectorStore
from transformers import CLIPModel, CLIPProcessor
from transformers import LlavaProcessor, LlavaForConditionalGeneration

import torch


class RAGChinBuilder:
    def __init__(self):
        # gpt-3.5-turbo 모델로 설정 (텍스트 입력만 처리)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")  # 텍스트 기반 GPT 사용
        self.vector_store = VectorStore()  # RAG 시스템의 VectorStore 사용
        self.retriever = self.vector_store.query_vector_store()

        # Hugging Face의 BLIP 모델 로드 (이미지 캡셔닝) - CPU로 설정
        self.device = torch.device("cpu")  # CPU에서 실행
        # CLIP 모델 및 프로세서 설정
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def load_image(self, image_path=None):
        """
        이미지 경로에서 이미지를 불러오거나, 이미지 URL에서 이미지를 불러옵니다.
        """
        if image_path:
            img = Image.open(image_path)  # 파일 시스템에서 이미지 로딩
        else:
            raise ValueError("이미지 경로 또는 URL을 제공하세요.")
        return img

    def get_image_embedding(self, img):
        """
        CLIP 모델을 사용하여 이미지 임베딩을 추출합니다.
        """
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        image_embedding = self.model.get_image_features(**inputs)  # 이미지에서 임베딩 추출
        return image_embedding

    def set_history_retriever(self):
        # 질문 재구성 프롬프트 템플릿 정의 및 생성과 히스토리 인식 검색기 생성
        contextualize_q_system_prompt = (
            "채팅 기록과 최신 사용자 질문이 주어졌을 때, "
            "해당 질문이 채팅 기록의 맥락을 참조할 수 있습니다. "
            "채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요. "
            "질문에 답변하지 말고, 필요한 경우에만 재구성하고 그렇지 않으면 그대로 반환하세요."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

        return history_aware_retriever

    def set_question_answer_chain(self):
        qa_system_prompt = (
            "당신은 의학적 질문 혹은 CT 사진을 받아서 병변을 진단한 후, 진단서를 작성하는 폐암 전문 보조의사입니다. "
            "검색된 문서와 제공된 CT 이미지의 내용을 사용하여 자세한 치료 계획과 함께 진단서를 작성하세요. "
            "추가적인 정보가 필요하다면, 진단을 내리지 말고 추가 질문을 하세요. "
            "답을 모르면 모른다고만 말하세요."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return question_answer_chain

    def get_rag_chain(self, image_path=None, prompt_text=None):
        # 이미지 로드: 이미지 경로에서 불러오기
        if image_path:
            img = self.load_image(image_path)
        else:
            raise ValueError("이미지를 제공해야 합니다.")

        if not prompt_text:
            prompt_text = "이 이미지는 폐암 진단을 위한 CT사진으로, 박스는 암을 찾은 것이다"

        # 이미지와 프롬프트를 결합한 캡션 생성 (CPU에서 처리)
        image_embedding = self.get_image_embedding(img)
        # GPT 모델을 사용해 텍스트 생성
        prompt_clip = f"Given the following image embedding: {image_embedding}, generate a detailed diagnosis."
        clip_result = self.llm(prompt_clip)
        print(clip_result)

        # 히스토리 기반 검색기 설정
        history_aware_retriever = self.set_history_retriever()

        # QA 체인 설정
        question_answer_chain = self.set_question_answer_chain()

        # RAG 체인 생성: 이미지 캡션과 텍스트 기반의 진단
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # RAG 체인 실행
        # result = rag_chain.invoke({"image_caption": image_caption})

        return rag_chain, clip_result
