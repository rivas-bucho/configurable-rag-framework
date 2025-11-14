# rag_system/chain.py

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 会話履歴ストアの定義 ---
# このストアはアプリケーションの実行中にメモリ上で会話履歴を保持する
# 本番環境では、Redisやデータベースなど永続的なストレージに置き換えることを検討する
store = {}


def get_session_history(session_id: str):
    """セッションIDに基づいて会話履歴を取得または新規作成する"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def format_docs(docs):
    """検索されたドキュメントのリストを、LLMに渡すための単一の文字列に整形する"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(llm, retriever, config):
    """
    設定に基づいてRAGチェーンを構築するメイン関数。
    会話履歴が有効かどうかで、構築するチェーンの種類を切り替える。
    """
    if config.memory.enable:
        return _build_conversational_chain(llm, retriever, config)
    else:
        return _build_standard_chain(llm, retriever, config)


def _build_standard_chain(llm, retriever, config):
    """
    会話履歴を考慮しない、一問一答形式のRAGチェーンを構築する。
    """
    # プロンプトテンプレートファイルを読み込む
    with open(config.prompt.template_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = ChatPromptTemplate.from_template(template)

    # LangChain Expression Language (LCEL) を用いてチェーンを定義
    chain = (
        RunnablePassthrough.assign(source_documents=retriever)
        .assign(context=lambda x: format_docs(x["source_documents"]))
        .assign(
            output=(
                # "context" と "question" を辞書としてプロンプトに渡す
                lambda x: {"context": x["context"], "question": x["question"]}
                | prompt
                | llm
                | StrOutputParser()  # LLMの出力を文字列に変換
            )
        )
    )
    return chain


def _build_conversational_chain(llm, retriever, config):
    """
    会話履歴を考慮する、対話形式のRAGチェーンを構築する。
    """
    # プロンプトテンプレートファイルを読み込む
    with open(config.prompt.chat_template_path, "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    # 会話履歴を差し込むためのプレースホルダーを含むプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # 基本的なRAGの流れを定義
    conversational_chain = (
        RunnablePassthrough.assign(
            source_documents=lambda x: retriever.invoke(x["question"])
        )
        .assign(context=lambda x: format_docs(x["source_documents"]))
        .assign(output=(prompt | llm | StrOutputParser()))
    )

    # 会話履歴管理機能でラップする
    return RunnableWithMessageHistory(
        conversational_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
