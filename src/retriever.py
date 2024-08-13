from langchain_community.vectorstores import FAISS

from src.vectordb import load_embed_model_from_local


def load_dense_retriever(doc_title, model_path, k):
    """Load the dense retriever"""
    embeddings = load_embed_model_from_local(model_path)
    vectorstore = FAISS.load_local(
        f"./faiss/{doc_title}", embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    return retriever
