from langchain_community.vectorstores import FAISS


def load_dense_retriever(doc_title, embed_model, k):
    """Load the dense retriever"""

    vectorstore = FAISS.load_local(
        f"./faiss/{doc_title}", embed_model, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    return retriever
