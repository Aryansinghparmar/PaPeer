from typing import Literal

from pydantic import BaseModel


class BtwRouteDecision(BaseModel):
    needs_web_search: bool


class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str
    # Added for claim verification; None defaults keep older constructions
    # valid and still produce a strict OpenAI schema (None defaults are stripped).
    evidence_id: str | None = None
    published: str | None = None
    source: Literal["arxiv", "web"] | None = None


ClaimStatus = Literal["superseded", "challenged", "still_holds", "insufficient_evidence"]


class ClaimVerificationResult(BaseModel):
    status: ClaimStatus
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]


class ClaimSearchPlan(BaseModel):
    claim: str
    arxiv_query: str
    web_query: str
