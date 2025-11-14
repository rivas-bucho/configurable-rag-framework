# rag_system/config.py

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

"""
このファイルは、Pydanticを使用して `config.yaml` の構造を型安全なクラスとして定義します。
これにより、設定ファイルのキーのタイプミスや値の型間違いをプログラム起動時に検知でき、
開発体験とシステムの堅牢性を向上させます。
各クラスが `config.yaml` の各セクションに対応しています。
"""


class DocumentLoaderConfig(BaseModel):
    """ドキュメントローダーの設定"""

    path: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RecursiveSplitterConfig(BaseModel):
    """再帰的テキスト分割（固定長）の設定"""

    chunk_size: int
    chunk_overlap: int


class SemanticSplitterConfig(BaseModel):
    """意味的テキスト分割の設定"""

    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile"
    ]
    breakpoint_threshold_amount: int


class TextSplitterConfig(BaseModel):
    """テキスト分割全体の設定"""

    splitter_type: Literal["semantic", "recursive"]
    recursive: RecursiveSplitterConfig
    semantic: SemanticSplitterConfig


class EmbeddingConfig(BaseModel):
    """埋め込みモデルの設定"""

    provider: Literal["google", "huggingface"]
    model_name: str


class VectorStoreConfig(BaseModel):
    """ベクトルストアの設定"""

    path: str


class RetrieverConfig(BaseModel):
    """Retriever（検索器）の設定"""

    enable_hybrid_search: bool
    hybrid_search_weights: List[float]


class RerankerConfig(BaseModel):
    """Re-ranker（再ランキング）の設定"""

    enable: bool
    provider: Literal["cohere"]
    top_n: int


class MemoryConfig(BaseModel):
    """会話履歴（メモリ）の設定"""

    enable: bool


class LLMConfig(BaseModel):
    """大規模言語モデル（LLM）の設定"""

    model_name: str
    temperature: float


class PromptConfig(BaseModel):
    """プロンプトテンプレートのパス設定"""

    template_path: str
    chat_template_path: str


class AppConfig(BaseModel):
    """
    アプリケーション全体のすべての設定を統括するトップレベルのクラス。
    `config.yaml` の内容全体がこのクラスのインスタンスにマッピングされる。
    """

    document_loader: DocumentLoaderConfig
    text_splitter: TextSplitterConfig
    embeddings: EmbeddingConfig
    vector_store: VectorStoreConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    memory: MemoryConfig
    llm: LLMConfig
    prompt: PromptConfig
