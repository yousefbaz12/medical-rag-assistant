from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from app.config import settings
from app.prompts import SYSTEM_PROMPT


EMERGENCY_KEYWORDS = [
    "severe chest pain",
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "shortness of breath",
    "passed out",
    "fainting",
    "fainted",
    "seizure",
    "coughing blood",
    "vomiting blood",
    "blue lips",
    "confusion",
    "stroke",
    "heart attack",
    "severe bleeding",
    "unconscious",
]


class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        self.vectorstore = FAISS.load_local(
            settings.vector_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "User question:\n{question}\n\nRetrieved context:\n{context}",
                ),
            ]
        )

    def retrieve(self, question: str, k: int | None = None):
        return self.vectorstore.similarity_search_with_score(
            question,
            k=k or settings.top_k,
        )

    def retrieve_debug(self, question: str, k: int | None = None) -> List[Dict[str, Any]]:
        results = self.retrieve(question, k=k)
        debug_items = []

        for doc, score in results:
            metadata = doc.metadata or {}
            debug_items.append(
                {
                    "score": float(score),
                    "source": metadata.get("source", "unknown"),
                    "focus": metadata.get("focus", "unknown"),
                    "related_question": metadata.get("question", "unknown"),
                    "content": doc.page_content,
                }
            )

        return debug_items

    def detect_safety_risk(self, question: str) -> Dict[str, Any]:
        lowered = question.lower()

        matched_keywords = [kw for kw in EMERGENCY_KEYWORDS if kw in lowered]

        if matched_keywords:
            return {
                "safety_alert": True,
                "safety_message": (
                    "This question may describe urgent symptoms. "
                    "The user should seek immediate medical attention or emergency care."
                ),
                "matched_keywords": matched_keywords,
            }

        return {
            "safety_alert": False,
            "safety_message": "",
            "matched_keywords": [],
        }

    def answer(self, question: str) -> Dict[str, Any]:
        results = self.retrieve(question)
        safety = self.detect_safety_risk(question)

        context_blocks = []
        sources = []

        for doc, score in results:
            metadata = doc.metadata or {}

            source_name = metadata.get("source", "unknown")
            focus = metadata.get("focus", "unknown")
            related_question = metadata.get("question", "unknown")

            context_blocks.append(
                f"Source: {source_name}\n"
                f"Focus: {focus}\n"
                f"Related Question: {related_question}\n"
                f"Knowledge: {doc.page_content}"
            )

            sources.append(
                {
                    "source": source_name,
                    "focus": focus,
                    "question": related_question,
                }
            )

        joined_context = "\n\n---\n\n".join(context_blocks)

        chain = self.prompt | self.llm
        response = chain.invoke(
            {
                "question": question,
                "context": joined_context,
            }
        )

        answer_text = response.content if hasattr(response, "content") else str(response)

        if safety["safety_alert"]:
            answer_text = (
                f"URGENT NOTICE: {safety['safety_message']}\n\n"
                f"{answer_text}"
            )

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_context": joined_context,
            "safety_alert": safety["safety_alert"],
            "safety_message": safety["safety_message"],
        }