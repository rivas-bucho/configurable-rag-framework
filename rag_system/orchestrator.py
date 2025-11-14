# rag_system/orchestrator.py

import logging

from .chain import build_rag_chain
from .components import ComponentFactory
from .config import AppConfig
from .retriever import build_retriever


class RAGOrchestrator:
    """
    RAGシステムの全てのコンポーネントを統括し、全体のワークフローを管理するクラス。
    システムのセットアップと、ユーザーからの質問への応答という2つの主要な責務を持つ。
    """

    def __init__(self, config: AppConfig):
        """
        Args:
            config (AppConfig): Pydanticで検証済みの設定オブジェクト
        """
        self.config = config
        self.rag_chain = None  # RAGチェーンはsetupメソッドで構築される

    def setup(self, force_recreate=False):
        """
        RAGシステムの全てのコンポーネントをセットアップする。
        このメソッドは、アプリケーション起動時に一度だけ呼び出されることを想定している。

        Args:
            force_recreate (bool): Trueの場合、既存のベクトルストアを削除して再構築する
        """
        logging.info("--- RAGシステムのセットアップを開始します ---")

        # 部品を生成するための工場（Factory）をインスタンス化
        factory = ComponentFactory(self.config)

        logging.info("1. 埋め込みモデルとLLMをロードしています...")
        embeddings = factory.create_embeddings()
        llm = factory.create_llm()

        logging.info("2. テキスト分割方法を準備しています...")
        # SemanticChunkerはembeddingsを必要とするため、ここでインスタンスを渡す
        text_splitter = factory.create_text_splitter(embeddings)

        logging.info("3. Re-rankerを準備しています...")
        reranker = factory.create_reranker()

        logging.info("4. 検索器(Retriever)を構築しています...")
        # データ準備の専門家（retriever.py）を呼び出す
        retriever = build_retriever(
            self.config, embeddings, text_splitter, reranker, force_recreate
        )

        logging.info("5. RAGチェーンを構築しています...")
        # チェーン構築の専門家（chain.py）を呼び出す
        self.rag_chain = build_rag_chain(llm, retriever, self.config)

        logging.info("--- RAGシステムのセットアップが完了しました ---")

    def ask(self, question: str, session_id: str = "default_session"):
        """
        ユーザーからの質問を受け取り、RAGチェーンを実行して回答を生成する。

        Args:
            question (str): ユーザーからの質問文
            session_id (str): 会話セッションを識別するためのID

        Returns:
            Tuple[str, List[Document]]: (回答文, 根拠となったドキュメントのリスト)
        """
        if not self.rag_chain:
            raise RuntimeError(
                "RAGチェーンがセットアップされていません。'setup()'メソッドを先に呼び出してください。"
            )

        # 入力形式を統一するために辞書型で渡す
        input_data = {"question": question}

        if self.config.memory.enable:
            # 会話履歴が有効な場合、セッションIDを設定に含める
            config = {"configurable": {"session_id": session_id}}
            result = self.rag_chain.invoke(input_data, config=config)
        else:
            # 会話履歴が無効な場合、単純にチェーンを実行
            result = self.rag_chain.invoke(input_data)

        return result.get("output", "回答がありません"), result.get(
            "source_documents", []
        )
