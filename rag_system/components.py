# rag_system/components.py

import os

from langchain_cohere import CohereRerank
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig

# --- プロバイダー名とLangChainクラスのマッピング ---
# 新しい埋め込みモデルプロバイダーを追加する際は、この辞書を編集するだけで対応できる
EMBEDDING_CLASSES = {
    "google": GoogleGenerativeAIEmbeddings,
    "huggingface": HuggingFaceEmbeddings,
}


class ComponentFactory:
    """
    設定ファイル（AppConfig）に基づいて、RAGシステムに必要なLangChainの
    コンポーネント（LLM, Embedding, TextSplitterなど）を生成するクラス。
    if文による分岐を減らし、拡張性を高める「ファクトリーパターン」を採用している。
    """

    def __init__(self, config: AppConfig):
        """
        Args:
            config (AppConfig): Pydanticで検証済みの設定オブジェクト
        """
        self.config = config

    def create_llm(self):
        """設定に基づいてLLMを生成する"""
        llm_config = self.config.llm
        return ChatGoogleGenerativeAI(
            model=llm_config.model_name, temperature=llm_config.temperature
        )

    def create_embeddings(self):
        """設定に基づいてEmbeddingモデルを生成する"""
        emb_config = self.config.embeddings
        provider = emb_config.provider

        if provider not in EMBEDDING_CLASSES:
            raise ValueError(
                f"サポートされていないEmbeddingsプロバイダーです: {provider}"
            )

        # マッピング辞書から対応するクラスを取得
        embedding_class = EMBEDDING_CLASSES[provider]

        # HuggingFaceEmbeddingsは引数名が 'model_name' なので特別扱いする
        if provider == "huggingface":
            return embedding_class(model_name=emb_config.model_name)
        # GoogleGenerativeAIEmbeddingsなどは引数名が 'model'
        return embedding_class(model=emb_config.model_name)

    def create_text_splitter(self, embeddings=None):
        """設定に基づいてText Splitterを生成する"""
        splitter_config = self.config.text_splitter
        if splitter_config.splitter_type == "semantic":
            # SemanticChunkerは埋め込みモデルを必要とするため、引数で受け取る
            if embeddings is None:
                raise ValueError(
                    "SemanticChunkerにはembeddingsインスタンスが必要です。"
                )
            semantic_conf = splitter_config.semantic
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=semantic_conf.breakpoint_threshold_type,
                breakpoint_threshold_amount=semantic_conf.breakpoint_threshold_amount,
            )
        else:  # "recursive"
            recursive_conf = splitter_config.recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=recursive_conf.chunk_size,
                chunk_overlap=recursive_conf.chunk_overlap,
            )

    def create_reranker(self):
        """設定に基づいてRe-rankerを生成する"""
        reranker_config = self.config.reranker
        if not reranker_config.enable:
            return None  # 無効の場合はNoneを返す

        if not os.getenv("COHERE_API_KEY"):
            raise ValueError(
                "Re-rankerが有効ですが、COHERE_API_KEYが.envファイルに設定されていません。"
            )

        # 現在はcohereのみ対応
        if reranker_config.provider == "cohere":
            return CohereRerank(top_n=reranker_config.top_n)

        return None
