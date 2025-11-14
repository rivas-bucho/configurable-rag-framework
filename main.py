# main.py

import argparse
import logging
import os

import yaml
from dotenv import load_dotenv

# rag_systemパッケージから、設定管理クラスとシステム全体を統括するクラスをインポート
from rag_system.config import AppConfig
from rag_system.orchestrator import RAGOrchestrator

# --- ロギングの基本設定 ---
# INFOレベル以上のログを、時刻、ログレベル、メッセージの形式でコンソールに出力する
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    プログラムのメイン実行関数。
    設定の読み込み、RAGシステムの初期化、ユーザーとの対話ループを管理する。
    """
    # .envファイルから環境変数（APIキーなど）を読み込む
    load_dotenv()

    # --- コマンドライン引数の設定 ---
    # --setup オプションを付けると、既存のベクトルストアを強制的に再構築できる
    parser = argparse.ArgumentParser(
        description="拡張可能なRAG（Retrieval-Augmented Generation）システム"
    )
    parser.add_argument(
        "--setup", action="store_true", help="ベクトルストアを強制的に再構築します。"
    )
    args = parser.parse_args()

    try:
        # --- 設定ファイルの読み込みと検証 ---
        logging.info("設定ファイル 'config.yaml' を読み込んでいます...")
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        # Pydanticモデルを使って、読み込んだ設定の型や構造を検証する
        config = AppConfig(**config_dict)
        logging.info("設定ファイルの読み込みと検証が完了しました。")

        # --- RAGシステムの準備 ---
        # 全体を統括するOrchestratorクラスをインスタンス化
        orchestrator = RAGOrchestrator(config)
        # RetrieverやChainなど、RAGに必要な全てのコンポーネントをセットアップ
        orchestrator.setup(force_recreate=args.setup)

    except Exception as e:
        # 初期化中に何らかのエラーが発生した場合は、ログに出力してプログラムを終了する
        logging.error(
            f"システムの初期化中に致命的なエラーが発生しました: {e}", exc_info=True
        )
        return

    # --- 対話インターフェースの開始 ---
    print("\n\n==================================================")
    print(" RAGシステム準備完了。質問を入力してください。")
    print(" ('exit' または 'quit' で終了します)")
    print("==================================================")

    # ユーザーセッションIDを仮に設定（WebアプリなどではユーザーごとにユニークなIDを割り当てる）
    session_id = "user_cli_session"
    while True:
        try:
            question = input("\n[質問]: ")
            if question.lower() in ["exit", "quit"]:
                print("システムを終了します。")
                break
            if not question.strip():  # 空の入力を無視
                continue

            # Orchestratorに質問を渡し、回答と根拠ドキュメントを受け取る
            answer, source_docs = orchestrator.ask(question, session_id)

            # --- 回答と根拠の表示 ---
            if answer:
                print("\n--- [回答] ---")
                print(answer)

            if source_docs:
                print("\n--- [回答の根拠となった情報] ---")
                for i, doc in enumerate(source_docs):
                    # コンテンツのプレビュー（長すぎる場合は省略）
                    content_preview = (
                        doc.page_content.strip().replace("\n", " ")[:150] + "..."
                    )
                    source_name = doc.metadata.get("source", "不明なソース")
                    print(
                        f"[{i + 1}] {content_preview} (Source: {os.path.basename(source_name)})"
                    )

        except KeyboardInterrupt:
            # Ctrl+Cが押されたら、きれいに終了する
            print("\nシステムを終了します。")
            break
        except Exception as e:
            # 対話ループ中のエラーはログに出力し、処理を続行する
            logging.error(f"質問応答中にエラーが発生しました: {e}", exc_info=True)


if __name__ == "__main__":
    main()
