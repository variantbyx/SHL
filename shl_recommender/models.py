from pydantic import BaseModel, Field
from typing import List, Optional, Any


class RecommendationRequest(BaseModel):
    query: Optional[str] = Field(None, description="Natural language query or job description text")
    job_description: Optional[str] = Field(None, description="Job description text (alias for query)")
    url: Optional[str] = Field(None, description="Optional URL to scrape job description from")
    top_k: Optional[int] = Field(10, ge=1, le=10, description="Number of recommendations (1 to 10)")
    balanced: Optional[bool] = Field(False, description="Whether to balance technical and soft skills")
    exclude_prepackaged: Optional[bool] = Field(False, description="Whether to exclude prepackaged job solutions")


class Assessment(BaseModel):
    url: str = Field(..., description="Valid URL to the assessment resource in SHL catalog")
    name: str = Field(..., description="Name of the assessment")
    adaptive_support: str = Field("No", description="Either 'Yes' or 'No' indicating if assessment supports adaptive testing")
    description: str = Field("", description="Detailed description of the assessment")
    duration: Optional[int] = Field(None, description="Duration of the assessment in minutes")
    remote_support: str = Field("Yes", description="Either 'Yes' or 'No' indicating if assessment can be taken remotely")
    test_type: List[str] = Field(default_factory=list, description="Categories or types of the assessment")


class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]
