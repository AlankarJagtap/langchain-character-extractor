# compute_embeddings.py

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load env variables from .env
load_dotenv()


def guess_story_title(text: str, fallback: str) -> str:
    """Use first non-empty line as title, else fallback."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return fallback


def load_story_documents(data_dir: str) -> list[Document]:
    """Load all story files in data_dir into LangChain Documents."""
    docs: list[Document] = []
    data_path = Path(data_dir)

    for file_path in data_path.glob("**/*"):
        if not file_path.is_file():
            continue

        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()

        story_title = guess_story_title(text, fallback=file_path.stem)

        metadata = {
            "source": str(file_path),
            "story_title": story_title,
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def compute_and_persist_embeddings(
    data_dir: str = "data",
    persist_dir: str = "chroma_db",
):
    """Main function to build embeddings and save vector store."""
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set. Put it in your .env file.")

    print(f"\n📘 Loading stories from: {data_dir}")
    docs = load_story_documents(data_dir)
    print(f"📄 Loaded {len(docs)} story files.")

    # -------- 1. Split into chunks --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )

    split_docs = splitter.split_documents(docs)
    print(f"✂️  Produced {len(split_docs)} raw chunks.")

    # -------- 2. Filter out empty chunks --------
    valid_docs = []
    for doc in split_docs:
        text = doc.page_content.strip()
        if text:  # keep only non-empty text
            valid_docs.append(doc)

    print(f"🧹 {len(valid_docs)} valid chunks after cleaning.")

    if not valid_docs:
        print("❌ No valid chunks to embed — exiting.")
        return

    # -------- 3. Embeddings --------
    print("🔢 Creating Mistral embedding model...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    print(f"💾 Creating Chroma DB at: {persist_dir}")
    vectorstore = Chroma.from_documents(
        documents=valid_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    vectorstore.persist()
    print("✅ Embeddings computed and stored successfully.\n")
