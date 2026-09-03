import os
import uvicorn
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import RecommendationRequest, RecommendationResponse
from recommender import recommend, recommend_balanced

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL individual assessment solutions based on job descriptions or queries.",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint required by SHL specification."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(payload: RecommendationRequest):
    """
    Assessment recommendation endpoint.
    Accepts a job description, natural language query, or URL and returns 5 to 10 relevant assessments.
    """
    extracted_text = None
    if payload.url:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(payload.url, timeout=10, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            text_candidates = []
            article = soup.find("article")
            if article:
                text_candidates.append(article.get_text(separator=" ").strip())

            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            if paragraphs:
                text_candidates.append(" \n".join(paragraphs))

            meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if meta and meta.get("content"):
                text_candidates.append(meta.get("content").strip())

            full_text = soup.get_text(separator=" ").strip()
            if full_text:
                text_candidates.append(full_text)

            text_candidates = [t for t in text_candidates if t]
            if text_candidates:
                extracted_text = max(text_candidates, key=len)
            else:
                raise ValueError("No extractable text found on the page")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {str(e)}")

    query_text = payload.query or payload.job_description or extracted_text
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text, job_description, or a valid URL must be provided.")

    top_k = payload.top_k or 10
    if top_k < 1:
        top_k = 10
    if top_k > 10:
        top_k = 10

    if payload.balanced:
        results = recommend_balanced(
            query_text,
            top_k=top_k,
            exclude_prepackaged=payload.exclude_prepackaged or False
        )
    else:
        results = recommend(
            query_text,
            top_k=top_k,
            exclude_prepackaged=payload.exclude_prepackaged or False
        )

    return {"recommended_assessments": results}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
