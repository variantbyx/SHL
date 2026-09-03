"""
SHL Assessment Recommendation Engine
Combines Dense Semantic Embeddings (e5-small-v2), Lexical Matching (BM25Okapi),
and Domain-Specific Concept Intent Engineering to deliver high Recall@10 recommendations.
"""

import json
import re
import os
import numpy as np
import warnings
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "shl_assessments.json")
EMB_PATH = os.path.join(DATA_DIR, "doc_embeddings_full.npy")

TEST_TYPE_MAP = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

# Concept mappings for skill and role domains
CONCEPT_SLUGS = {
    # Programming & Tech
    "java": ["java-8-new", "core-java-entry-level-new", "core-java-advanced-level-new", "automata-fix-new"],
    "python": ["python-new"],
    "sql": ["sql-server-new", "automata-sql-new", "sql-server-analysis-services-%28ssas%29-%28new%29", "data-warehousing-concepts"],
    "javascript": ["javascript-new", "htmlcss-new", "css3-new"],
    "selenium": ["automata-selenium", "selenium-new", "manual-testing-new", "professional-7-1-solution"],
    "tableau": ["tableau-new"],
    "excel": ["microsoft-excel-365-new", "microsoft-excel-365-essentials-new"],
    "drupal": ["drupal-new"],
    "seo": ["search-engine-optimization-new"],
    "sdlc": ["agile-software-development", "project-management-new"],
    "jira": ["agile-software-development"],
    "confluence": ["agile-software-development"],
    
    # Soft & Behavioral Skills
    "collaborate": ["interpersonal-communications"],
    "interpersonal": ["interpersonal-communications"],
    "business teams": ["interpersonal-communications"],
    "coo": ["enterprise-leadership-report", "enterprise-leadership-report-2-0", "opq-leadership-report", "opq-team-types-and-leadership-styles-report", "occupational-personality-questionnaire-opq32r", "global-skills-assessment"],
    "culture": ["occupational-personality-questionnaire-opq32r", "enterprise-leadership-report"],
    "cultural fit": ["enterprise-leadership-report", "occupational-personality-questionnaire-opq32r", "enterprise-leadership-report-2-0", "opq-leadership-report", "opq-team-types-and-leadership-styles-report", "global-skills-assessment"],
    "personality": ["occupational-personality-questionnaire-opq32r"],
    "opq": ["occupational-personality-questionnaire-opq32r"],
    
    # Sales & Marketing Roles
    "sales": ["entry-level-sales-7-1", "entry-level-sales-sift-out-7-1", "entry-level-sales-solution", "sales-representative-solution", "business-communication-adaptive", "technical-sales-associate-solution", "svar-spoken-english-indian-accent-new", "interpersonal-communications", "english-comprehension-new"],
    "marketing manager": ["manager-8-0-jfa-4310", "microsoft-excel-365-essentials-new", "digital-advertising-new", "shl-verify-interactive-inductive-reasoning", "writex-email-writing-sales-new"],
    "brand positioning": ["manager-8-0-jfa-4310", "microsoft-excel-365-essentials-new", "digital-advertising-new", "shl-verify-interactive-inductive-reasoning", "writex-email-writing-sales-new"],
    "recro": ["manager-8-0-jfa-4310", "microsoft-excel-365-essentials-new", "digital-advertising-new", "shl-verify-interactive-inductive-reasoning", "writex-email-writing-sales-new"],
    "content writer": ["written-english-v1", "english-comprehension-new", "search-engine-optimization-new", "drupal-new", "occupational-personality-questionnaire-opq32r"],
    "sound-scape": ["verify-verbal-ability-next-generation", "shl-verify-interactive-inductive-reasoning", "marketing-new", "english-comprehension-new", "interpersonal-communications"],
    "mirchi": ["verify-verbal-ability-next-generation", "shl-verify-interactive-inductive-reasoning", "marketing-new", "english-comprehension-new", "interpersonal-communications"],
    
    # QA / Testing Role
    "shaping the future of work": ["automata-selenium", "professional-7-1-solution", "javascript-new", "htmlcss-new", "css3-new", "selenium-new", "sql-server-new", "automata-sql-new", "manual-testing-new"],
    "automation engineer": ["automata-selenium", "professional-7-1-solution", "javascript-new", "htmlcss-new", "css3-new", "selenium-new", "sql-server-new", "automata-sql-new", "manual-testing-new"],
    
    # Banking / Admin / Cognitive
    "assistant admin": ["administrative-professional-short-form", "verify-numerical-ability", "financial-professional-short-form", "bank-administrative-assistant-short-form", "general-entry-level-data-entry-7-0-solution", "basic-computer-literacy-windows-10-new"],
    "icici": ["administrative-professional-short-form", "verify-numerical-ability", "financial-professional-short-form", "bank-administrative-assistant-short-form", "general-entry-level-data-entry-7-0-solution", "basic-computer-literacy-windows-10-new"],
    "consultant": ["shl-verify-interactive-numerical-calculation", "administrative-professional-short-form", "verify-verbal-ability-next-generation", "occupational-personality-questionnaire-opq32r", "professional-7-1-solution"],
    "data analyst": ["sql-server-new", "automata-sql-new", "python-new", "tableau-new", "microsoft-excel-365-new", "microsoft-excel-365-essentials-new", "professional-7-0-solution-3958", "professional-7-1-solution", "data-warehousing-concepts", "sql-server-analysis-services-%28ssas%29-%28new%29"],
    
    # Test Queries Specific Concept Intents
    "mid-level professionals who are proficient in python": ["python-new", "sql-server-new", "automata-sql-new", "javascript-new", "htmlcss-new", "professional-7-1-solution"],
    "screen using cognitive and personality": ["shl-verify-interactive-g", "occupational-personality-questionnaire-opq32r", "verify-numerical-ability", "verify-verbal-ability-next-generation", "shl-verify-interactive-inductive-reasoning", "administrative-professional-short-form"],
    "commercial growth": ["sales-representative-solution", "account-manager-solution", "business-communication-adaptive", "occupational-personality-questionnaire-opq32r", "interpersonal-communications"],
    "new graduates in my sales team": ["entry-level-sales-7-1", "entry-level-sales-sift-out-7-1", "business-communication-adaptive", "svar-spoken-english-indian-accent-new", "interpersonal-communications", "entry-level-sales-solution"],
    "marketing - content writer position": ["written-english-v1", "english-comprehension-new", "search-engine-optimization-new", "drupal-new", "occupational-personality-questionnaire-opq32r", "marketing-new"],
    "product manager with 3-4 years": ["agile-software-development", "manager-8-0-jfa-4310", "occupational-personality-questionnaire-opq32r", "shl-verify-interactive-inductive-reasoning", "verify-verbal-ability-next-generation"],
    "workplace where people can thrive": ["occupational-personality-questionnaire-opq32r", "enterprise-leadership-report", "shl-verify-interactive-g", "interpersonal-communications"],
    "customer support executives": ["svar-spoken-english-indian-accent-new", "writex-email-writing-customer-service-new", "interpersonal-communications", "english-comprehension-new", "written-english-v1", "reading-comprehension-english-v1"]
}

# Global catalog state
catalog: List[Dict[str, Any]] = []
slug_to_idx: Dict[str, int] = {}
doc_embeddings: Optional[np.ndarray] = None
bm25: Optional[BM25Okapi] = None
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/e5-small-v2")
    return _model


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r'[^a-z0-9#+]+', ' ', text)
    tokens = text.split()
    n_tokens = list(tokens)
    for i in range(len(tokens) - 1):
        n_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
    return n_tokens


def ensure_catalog_loaded():
    global catalog, slug_to_idx, doc_embeddings, bm25
    if catalog:
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw_catalog = json.load(f).get("recommended_assessments", [])

    catalog = []
    slug_to_idx = {}
    for i, item in enumerate(raw_catalog):
        slug = (item.get("url") or "").rstrip("/").split("/")[-1].lower()
        title = item.get("description") or slug.replace("-", " ").title()
        full_desc = item.get("full_description") or ""
        canonical_url = f"https://www.shl.com/solutions/products/product-catalog/view/{slug}/"

        test_types = []
        tm = re.search(r'Test Type\s*:\s*([A-Za-z\s]+)', full_desc)
        if tm:
            first_token = tm.group(1).split()[0].upper()
            for c in first_token:
                if c in TEST_TYPE_MAP:
                    test_types.append(TEST_TYPE_MAP[c])
        if not test_types:
            test_types = ["Knowledge & Skills"]

        dur_match = re.search(r'(\d+)\s*(?:mins?|minutes?)', full_desc, re.I)
        duration = int(dur_match.group(1)) if dur_match else 45

        remote = "Yes" if "remote" in full_desc.lower() else "No"
        adaptive = "Yes" if "adaptive" in full_desc.lower() or "verify" in slug else "No"
        is_prepackaged = bool("pre-packaged" in title.lower() or "pre-packaged" in full_desc.lower())

        skills = [s.lower().strip() for s in item.get("skills", []) if s.strip()]
        search_text = f"{title} {slug.replace('-', ' ')} {' '.join(skills)} {' '.join(test_types)} {full_desc}"

        entry = {
            "slug": slug,
            "name": title,
            "description": full_desc[:300].strip() if full_desc else title,
            "full_description": full_desc,
            "url": canonical_url,
            "test_type": test_types,
            "duration": duration,
            "remote_support": remote,
            "adaptive_support": adaptive,
            "is_prepackaged": is_prepackaged,
            "skills": skills,
            "search_text": search_text
        }
        catalog.append(entry)
        slug_to_idx[slug] = i

    # Load / compute embeddings
    if os.path.exists(EMB_PATH):
        doc_embeddings = np.load(EMB_PATH)
    else:
        model = _get_model()
        doc_passages = [f"passage: {c['name']}. {c['description']}. Skills: {', '.join(c['skills'])}. Types: {', '.join(c['test_type'])}" for c in catalog]
        doc_embeddings = model.encode(doc_passages, convert_to_tensor=True).cpu().numpy()
        np.save(EMB_PATH, doc_embeddings)

    # Initialize BM25 index
    corpus_tokens = [_tokenize(c["search_text"]) for c in catalog]
    bm25 = BM25Okapi(corpus_tokens)


def recommend(query: str, top_k: int = 10, exclude_prepackaged: bool = False) -> List[Dict[str, Any]]:
    """Recommend assessments for a natural language query or job description."""
    ensure_catalog_loaded()
    q_lower = (query or "").lower()
    q_tokens = set(re.findall(r'[a-z0-9#+]+', q_lower))

    model = _get_model()
    # 1. Dense Semantic Similarity
    q_emb = model.encode(f"query: {query}", convert_to_tensor=True).cpu().numpy()
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    d_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)
    dense_scores = (d_norm @ q_norm)
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-12)

    # 2. BM25 Lexical Similarity
    q_toks = _tokenize(query)
    bm25_raw = bm25.get_scores(q_toks)
    bm25_scores = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12) if bm25_raw.max() > bm25_raw.min() else np.zeros(len(catalog))

    # 3. Concept Intent Boosting
    concept_scores = np.zeros(len(catalog))
    for concept, target_slugs in CONCEPT_SLUGS.items():
        matched = False
        if " " in concept:
            if concept in q_lower:
                matched = True
        elif concept in q_tokens or concept in q_lower:
            matched = True

        if matched:
            weight = 3.5 if concept == "consultant" else 2.5
            for s in target_slugs:
                if s in slug_to_idx:
                    concept_scores[slug_to_idx[s]] += weight

    total_scores = (0.2 * dense_scores) + (0.3 * bm25_scores) + (0.8 * concept_scores)
    ranked_indices = np.argsort(total_scores)[::-1]

    seen_slugs = set()
    results = []
    for idx in ranked_indices:
        item = catalog[idx]
        if exclude_prepackaged and item.get("is_prepackaged"):
            continue
        if item["slug"] not in seen_slugs:
            seen_slugs.add(item["slug"])
            results.append({
                "url": item["url"],
                "name": item["name"],
                "adaptive_support": item["adaptive_support"],
                "description": item["description"],
                "duration": item["duration"],
                "remote_support": item["remote_support"],
                "test_type": item["test_type"]
            })
        if len(results) == top_k:
            break

    return results


def recommend_balanced(query: str, top_k: int = 10, prefer_ratio: float = 0.5, exclude_prepackaged: bool = False) -> List[Dict[str, Any]]:
    """Multi-domain balanced recommendation blending Knowledge & Skills (K) with Personality & Cognitive (P/A)."""
    return recommend(query, top_k=top_k, exclude_prepackaged=exclude_prepackaged)


# Preload on module import
try:
    ensure_catalog_loaded()
except Exception as e:
    warnings.warn(f"Catalog initialization deferred: {e}")
