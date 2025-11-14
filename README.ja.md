# 汎用RAG基盤 (Configurable RAG Framework)

これは、設定ファイル一つでコンポーネントを自由に着せ替えできる、拡張性の高いRetrieval-Augmented Generation (RAG) システムのテンプレートです。

## ✨ 特徴

- **設定駆動**: `config.yaml` を編集するだけで、コードを触らずにシステムの挙動を変更できます。
- **コンポーネント指向**: LLM、埋め込みモデル、Retrieverなどを柔軟に差し替え可能です。
- **高機能**: ハイブリッド検索、Re-ranking、会話履歴など、高度な機能を設定でON/OFFできます。
- **拡張性**: 新しいモデルやコンポーネントを簡単に追加できる設計になっています。

## 🔧 動作要件

- Python 3.9 以上
- APIキー（Google AI, Cohereなど）

## 🚀 セットアップ方法

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/あなたのユーザー名/あなたのリポジトリ名.git
    cd あなたのリポジトリ名
    ```

2.  **仮想環境の作成と有効化 (推奨):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    .\venv\Scripts\activate   # Windows
    ```

3.  **必要なライブラリをインストール:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **.envファイルの作成:**
    `.env.example` ファイルをコピーして `.env` という名前のファイルを作成し、あなたのAPIキーを記述してください。
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    COHERE_API_KEY="your_cohere_api_key_here"
    ```

5.  **知識源ドキュメントの配置:**
    `knowledge_docs/` ディレクトリに、知識源としたいテキストファイルやPDFなどを配置してください。

6.  **ベクトルストアの構築:**
    `--setup` オプションを付けて初回実行します。これにより、ドキュメントが読み込まれ、ベクトル化されます。
    ```bash
    python main.py --setup
    ```

## 使い方

セットアップが完了したら、以下のコマンドで対話を開始できます。

```bash
python main.py