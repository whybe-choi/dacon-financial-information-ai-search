import re


def format_docs(docs):
    """Format retrieved documents"""
    context = ""
    for doc in docs:
        context += doc.page_content + "\n-----\n"
    return context


def postprocess_answer(text):
    """Postprocess answer text"""
    # 들여쓰기를 띄어쓰기로 변환
    processed_text = text.replace("\n", " ")

    # 여러 공백을 하나의 공백으로 변환
    processed_text = re.sub(r"\s+", " ", processed_text)

    # 문자열 양쪽 끝의 공백 제거
    processed_text = processed_text.strip()

    return processed_text
