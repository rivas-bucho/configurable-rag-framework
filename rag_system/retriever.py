# rag_system/retriever.py

import logging
import os
import shutil

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS


def load_documents(config):
    """
    設定に基づいて、ローカルファイルとWeb URLからドキュメントを読み込む。
    """
    doc_config = config.document_loader
    documents = []

    # ローカルドキュメントの読み込み
    if doc_config.path and os.path.exists(doc_config.path):
        logging.info(
            f"ローカルドキュメントを '{doc_config.path}' から読み込んでいます..."
        )
        loader = DirectoryLoader(
            doc_config.path,
            glob="**/*.*",
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={"autodetect_encoding": True},
        )
        documents.extend(loader.load())

    # Webページの読み込み
    if doc_config.urls:
        logging.info(f"Webページを {len(doc_config.urls)} 件読み込んでいます...")
        loader = WebBaseLoader(doc_config.urls)
        documents.extend(loader.load())

    if not documents:
        raise ValueError(
            "読み込むドキュメントが見つかりませんでした。パスやURLの設定を確認してください。"
        )
    return documents


def build_retriever(config, embeddings, text_splitter, reranker, force_recreate=False):
    """
    Retriever（検索器）を構築するメイン関数。
    ベクトルストアの新規作成または読み込み、ハイブリッド検索、Re-rankerの適用を行う。

    Args:
        config (AppConfig): アプリケーション全体の設定
        embeddings: 埋め込みモデルのインスタンス
        text_splitter: テキスト分割器のインスタンス
        reranker: Re-rankerのインスタンス（またはNone）
        force_recreate (bool): Trueの場合、既存のベクトルストアを削除して再構築する

    Returns:
        BaseRetriever: 最終的に構築されたRetrieverインスタンス
    """
    vs_path = config.vector_store.path

    # --setup オプションが指定された場合、既存のストアを削除
    if force_recreate and os.path.exists(vs_path):
        logging.info(f"既存のベクトルストア '{vs_path}' を削除しています...")
        shutil.rmtree(vs_path)

    if not os.path.exists(vs_path):
        # --- ベクトルストアの新規作成 ---
        logging.info(f"ベクトルストアを新規に構築しています... (保存先: {vs_path})")
        docs = load_documents(config)

        logging.info("テキストをチャンクに分割しています...")
        texts = text_splitter.split_documents(docs)
        logging.info(f"ドキュメントを {len(texts)} 個のチャンクに分割しました。")

        logging.info("チャンクをベクトル化し、FAISSインデックスを構築しています...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(vs_path)
        logging.info("ベクトルストアの構築と保存が完了しました。")

        # --- ベースとなるRetrieverの構築 ---
        faiss_retriever = vectorstore.as_retriever()
        if config.retriever.enable_hybrid_search:
            logging.info(
                "ハイブリッド検索（意味検索 + キーワード検索）を有効化します。"
            )
            bm25_retriever = BM25Retriever.from_documents(texts)
            base_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=config.retriever.hybrid_search_weights,
            )
        else:
            base_retriever = faiss_retriever
    else:
        # --- 既存ベクトルストアの読み込み ---
        logging.info(f"既存のベクトルストア '{vs_path}' から検索器を構築しています...")
        vectorstore = FAISS.load_local(
            vs_path, embeddings, allow_dangerous_deserialization=True
        )
        base_retriever = vectorstore.as_retriever()

    # --- Re-rankerの適用 ---
    # rerankerが有効な場合、ベースのRetrieverをラップして精度を向上させる
    if reranker:
        logging.info("Cohere Re-rankerを有効化し、検索結果を再ランキングします。")
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

    return base_retriever
