# Configurable RAG Framework

This repository provides a highly extensible template for a Retrieval-Augmented Generation (RAG) system, designed to be easily configured by a single `config.yaml` file.

## ✨ Features

- **Configuration-Driven**: Modify the system's behavior, such as switching models or toggling features, without changing any code.
- **Modular by Design**: Easily swap out components like LLMs, embedding models, and retrievers.
- **Advanced Features**: Built-in support for hybrid search, re-ranking, and conversational memory, all switchable via config.
- **Extensible**: The architecture makes it simple to add new components and providers.

## 🔧 Requirements

- Python 3.9+
- API Keys (e.g., Google AI, Cohere)

## 🚀 Getting Started

Follow these steps to set up and run the project.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rivas-bucho/configurable-rag-framework.git
    cd configurable-rag-framework
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Copy the example file and fill in your API keys.
    ```bash
    cp .env.example .env
    # Now, edit the .env file with your actual keys.
    ```

5.  **Place your knowledge documents:**
    Add your text files, PDFs, or other documents into the `knowledge_docs/` directory.

6.  **Build the vector store:**
    Run the script with the `--setup` flag for the first time. This will process your documents and create a vector index.
    ```bash
    python main.py --setup
    ```

## 💻 Usage

Once the setup is complete, you can start the interactive chat with:

```bash
python main.py