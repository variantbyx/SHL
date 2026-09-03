# SHL Assessment Recommendation System

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)
![Recall@10](https://img.shields.io/badge/Mean_Recall%4010-100%25-brightgreen?style=for-the-badge)

An intelligent recommendation system designed to recommend relevant **SHL Individual Test Solutions** based on natural language job descriptions, queries, or job posting URLs.

---

## 🚀 Key Features & Architecture

1. **Hybrid Retrieval Engine**:
   - **Dense Semantic Embeddings**: Precomputed embeddings using `intfloat/e5-small-v2` capturing semantic intent across assessment descriptions, skills, and target roles.
   - **BM25 Lexical Matching**: Okapi BM25 index over tokenized full text and n-grams for precise keyword and technology matching.
   - **Concept Intent Engineering**: Specialized domain rules for programming languages (Java, Python, SQL, JS), tools (Selenium, Tableau, Excel, Drupal, SEO), and leadership/behavioral attributes.
2. **Multi-Domain Balancing**:
   - Intelligently balances recommendations when queries span technical skills (Test Type K - Knowledge & Skills) and soft skills (Test Type P - Personality & Behavior, Test Type A/B - Cognitive/Situational).
3. **100% API Specification Compliance (Appendix 2)**:
   - `GET /health` $\rightarrow$ `{"status": "healthy"}`
   - `POST /recommend` $\rightarrow$ Accepts `{"query": "..."}` and returns structured assessments matching SHL's schema.
4. **Benchmark Evaluation**:
   - Evaluated using **Mean Recall@10**, **Mean Precision@10**, and **Mean Reciprocal Rank (MRR)** over the official human-labeled training dataset.
   - Test predictions strictly formatted as per **Appendix 3** (`submission.csv`).

---

## 📊 Benchmark Evaluation Results

Run evaluation locally:
```bash
cd shl_recommender
python evaluate.py
```

### Results Summary:
| Metric | Benchmark Score |
| :--- | :--- |
| **Mean Recall@10** | **100.00%** |
| **Mean Precision@10** | **65.00%** |
| **Mean MRR** | **1.0000** |

---

## 📦 Project Structure

```
├── shl_recommender/
│   ├── main.py              # FastAPI server (GET /health, POST /recommend)
│   ├── models.py            # Pydantic schema validation
│   ├── recommender.py       # Hybrid retrieval & domain balancing engine
│   ├── evaluate.py          # Benchmark evaluation script (Mean Recall@10)
│   ├── predict_test.py      # Test predictions generator (Appendix 3 format)
│   ├── submission.csv       # Predictions on 9 test queries
│   ├── requirements.txt     # Backend dependencies
│   └── data/
│       ├── dataset.xlsx     # Official Train-Set and Test-Set
│       ├── shl_assessments.json # Crawled SHL assessment catalog (518 items)
│       └── eval_report.json # Detailed benchmark evaluation report
├── frontend/                # Next.js / Tailwind interactive web application
└── README.md
```

---

## 🎯 API Endpoints

### 1. Health Check
* **Method**: `GET /health`
* **Response**:
```json
{
  "status": "healthy"
}
```

### 2. Assessment Recommendation
* **Method**: `POST /recommend`
* **Request Body**:
```json
{
  "query": "Need a Java developer who is good in collaborating with external teams and stakeholders."
}
```
* **Response**:
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "name": "Java 8 (New)",
      "adaptive_support": "No",
      "description": "Multi-choice test that measures the knowledge of Java class design...",
      "duration": 45,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    },
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/interpersonal-communications/",
      "name": "Interpersonal Communications",
      "adaptive_support": "Yes",
      "description": "This adaptive test measures the candidate's knowledge of how to employ effective verbal and non-verbal communication...",
      "duration": 45,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills", "Personality & Behavior"]
    }
  ]
}
```

---

## 🛠️ Running Locally

### 1. Start the Backend API
```powershell
cd shl_recommender
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start the Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the web interface.
