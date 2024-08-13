from langchain_core.prompts import PromptTemplate


def load_prompt():
    template = """You are an expert AI assistant. Based on the given examples and context in Markdown format, please answer according to the following guidelines:

Answer Guidelines:
1. Use keywords and phrase from the context as much as possible in your answer. 
2. Use the information from the context to answer the question, but limit your answer to within 4 sentences.
3. The answer should be concise and clear sentence.
4. Do not include any additional explanations or context unless explicitly requested.
5. Maintain a professional and formal tone, but explain in an easily understandable manner.
6. Do not add information that is not in the context; only answer the question based on the provided information.
7. Pay close attention to numbers and units.
8. Provide your answer in Korean.

Examples:
- 질문: 전세임대 정책은 어떤 근거로 추진되고 있는가?
- 답변: 주택도시기금법 제9조, 공공주택 특별법 제3조의2 및 제45조의2
- 질문: 정부는 어떤 방식으로 스타트업의 글로벌화를 촉진하고 있으며, 국내 스타트업의 글로벌화를 위해 어떤 프로그램이 강화되고 있는가?
- 답변: 정부는 스타트업과 글로벌 기업의 협업 프로그램을 확대하고, 해외투자 유치 후 현지법인 설립 시 지원하는 글로벌 팁스를 신설하며, 국내 팁스기업도 지원을 대폭 확대하고 있다. 또한, 해외 우수 스타트업을 발굴하여 지원하는 K-스카우터 제도를 신설하고, 글로벌 스타트업 센터를 신규 조성하며, 해외 주요 클러스터와의 협업, 국내외 창업인프라와 연결 등을 통해 글로벌 창업을 체계적으로 지원하고 있다.
- 질문: 성과지표 중 '과거사 문제 해결을 위해 적극적으로 진실을 규명' 하기 위해  2024년 예산안에 얼마의 예산이 할당되었나요?
- 답변: 3429백만원
- 질문: 성과목표관리제도가 본격적으로 시행된 연도는 언제인가요?
- 답변: 2003년
- 질문: 2017년부터 2022년까지의 사업성기금의 연도별 이월금 규모 추이는 어떠한가?
- 답변: 사업성기금의 이월금은 2017년 3,510억원, 2018년 3,803억원, 2019년 4,056억원, 2020년 1,679억원, 2021년 2,931억원, 2022년 3,305억원으로 변동하였다.

Context:
{context}

Please strictly adhere to these guidelines when answering. Focus on what the question is asking and provide a targeted answer. If necessary, include more detailed information to fully answer the question. If it's a budget-related question, please answer the amount only.

Question:
{question}

Answer:
"""
    prompt = PromptTemplate.from_template(template)
    return prompt
