import os
import unicodedata

import pandas as pd
import pymupdf4llm
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer


def save_embed_model_to_local(model_id, model_path):
    """Save Embedding Model to Local"""
    model = SentenceTransformer(model_id)
    model.save(model_path)


def load_embed_model_from_local(model_path):
    """Load Embedding Model from Local"""
    embedding = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding


def load_and_split_docs(file_path, tokenizer):
    """Load PDF file and split"""
    markdown_text = pymupdf4llm.to_markdown(file_path)
    page_contents = markdown_text.split("\n-----\n")[:-1]
    documents = [Document(page_content=page_content) for page_content in page_contents]

    print(f"문서의 총 페이지 수: {len(page_contents)}")
    print(f"문서의 총 토큰 수: {len(tokenizer.encode(markdown_text))}")

    return documents


def create_vectorstore(chunks, model_path, title):
    """Create vectorstore"""
    if not os.path.exists("faiss"):
        os.makedirs("faiss")

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=load_embed_model_from_local(model_path),
    )

    vector_db.save_local(f"./faiss/{title}")


def process_pdfs_from_dataframe(df, model_path, tokenizer):
    "Store DB using PDF names"
    sources = [unicodedata.normalize("NFC", source) for source in df["Source"].unique()]

    source_mapping = {}
    for idx, source in enumerate(sources):
        source_mapping[source] = f"test{idx+1}"

    # Rename PDF files
    test_file_paths = os.listdir("./data/test_source")

    for path in test_file_paths:
        old_path = os.path.join("./data/test_source", path)
        source = unicodedata.normalize("NFC", path.rstrip(".pdf"))
        new_path = os.path.join("./data/test_source", f"{source_mapping[source]}.pdf")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

    for source in tqdm(sources, desc="Processing PDFs"):
        file_path = f"./data/test_source/{source_mapping[source]}.pdf"

        print(f"Processing {source}...")

        chunks = load_and_split_docs(file_path, tokenizer)
        create_vectorstore(chunks, model_path, source_mapping[source])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("rtzr/ko-gemma-2-9b-it")

    if not os.path.exists("models"):
        os.makedirs("models")
    save_embed_model_to_local("BAAI/bge-m3", "./models/bge-m3")

    df = pd.read_csv("./data/test.csv")
    process_pdfs_from_dataframe(df, "./models/bge-m3", tokenizer)
