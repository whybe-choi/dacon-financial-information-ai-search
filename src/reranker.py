from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def load_reranker(retriever, top_n=3):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever
