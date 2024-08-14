# dacon-financial-information-ai-search
데이콘 재정정보 AI 검색 알고리즘 경진대회

## How to use
1. Clone the repository
```bash
git clone https://github.com/whybe-choi/dacon-financial-information-ai-search.git
cd dacon-financial-information-ai-search
```

2. Create a virtual environment & Install the dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Save FAISS to local
```bash
python src/vectordb.py
```

4. Inference
```bash
python main.py
```

## Directory Structure
```
.
├── LICENSE
├── README.md
├── data
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── test_source
│   ├── train.csv
│   └── train_source
├── faiss
│   └── ...
├── main.py
├── models
│   └── ...
├── requirements.txt
└── src
    ├── llm.py
    ├── prompt.py
    ├── reranker.py
    ├── retriever.py
    ├── utils.py
    └── vectordb.py
```