import unicodedata

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm

from src.llm import load_pipeline
from src.reranker import load_reranker
from src.retriever import load_dense_retriever
from src.utils import format_docs, postprocess_answer
from src.vectordb import load_embed_model_from_local
from src.prompt import load_prompt


def main():
    llm = load_pipeline("rtzr/ko-gemma-2-9b-it")
    embed_model = load_embed_model_from_local("./models/bge-m3")

    df = pd.read_csv("./data/test.csv")
    df["Source"] = df["Source"].apply(lambda row: unicodedata.normalize("NFC", row))

    source_mapping = {}
    for idx, source in enumerate(df["Source"].unique()):
        source_mapping[source] = f"test{idx+1}"

    retrievers = {}
    for source in df["Source"].unique():
        retriever = load_dense_retriever(source_mapping[source], embed_model, 10)
        retrievers[source] = load_reranker(retriever, top_n=2)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering Questions"):
        source = row["Source"]
        question = row["Question"]

        prompt = load_prompt()

        chain = (
            {
                "context": retrievers[source] | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        response = postprocess_answer(response)

        print(f"Question : {question}")
        print(f"Answer : {response}\n")

        # 결과 저장
        results.append(
            {
                "Source": row["Source"],
                "Source_path": row["Source_path"],
                "Question": question,
                "Answer": response,
            }
        )

    # 답안 제출
    submit_df = pd.read_csv("./data/sample_submission.csv")
    submit_df["Answer"] = [item["Answer"] for item in results]
    submit_df.to_csv("./submission.csv", encoding="UTF-8-sig", index=False)


if __name__ == "__main__":
    main()
