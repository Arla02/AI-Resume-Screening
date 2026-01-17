"""Pydantic models for structured data throughout the screening pipeline."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Resume Data Models
# ============================================================================

class ContactInfo(BaseModel):
    """Candidate contact information."""
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""


class Education(BaseModel):
    """Educational background entry."""
    degree: str = ""
    field: str = ""
    institution: str = ""
    graduation_year: str = ""
    gpa: str = ""


class WorkExperience(BaseModel):
    """Work experience entry."""
    title: str
    company: str
    duration: str = ""
    start_date: str = ""
    end_date: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class Skill(BaseModel):
    """Extracted skill with metadata."""
    name: str
    category: Literal["technical", "soft_skill", "tool", "language", "framework", "other"] = "other"
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"] = "intermediate"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    source: Literal["explicit", "inferred"] = "explicit"


class ResumeData(BaseModel):
    """Structured representation of a parsed resume."""
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    skills_section: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    raw_text: str = ""
    parsing_confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    parsing_notes: list[str] = Field(default_factory=list)


# ============================================================================
# Job Analysis Models
# ============================================================================

class Requirement(BaseModel):
    """A single job requirement."""
    description: str
    category: Literal["skill", "experience", "education", "certification", "other"] = "other"
    priority: Literal["required", "preferred", "nice_to_have"] = "required"
    years_needed: Optional[int] = None


class JobRequirements(BaseModel):
    """Structured representation of job requirements."""
    title: str = ""
    summary: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: int = 0
    education_requirements: list[str] = Field(default_factory=list)
    certifications_required: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    parsing_confidence: float = Field(ge=0.0, le=1.0, default=0.8)


# ============================================================================
# Matching & Evaluation Models
# ============================================================================

class SkillMatch(BaseModel):
    """Result of matching a single skill requirement."""
    requirement: str
    matched: bool
    matched_skill: str = ""
    match_quality: Literal["exact", "semantic", "partial", "none"] = "none"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    notes: str = ""


class SkillsMatchResult(BaseModel):
    """Overall skills matching result."""
    matches: list[SkillMatch] = Field(default_factory=list)
    required_skills_met: int = 0
    required_skills_total: int = 0
    preferred_skills_met: int = 0
    preferred_skills_total: int = 0
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    reasoning: str = ""


class ExperienceEvaluation(BaseModel):
    """Experience evaluation result."""
    years_relevant: float = 0.0
    years_required: int = 0
    experience_score: float = Field(ge=0.0, le=1.0, default=0.0)
    role_relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    career_progression: str = ""
    gaps_identified: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    reasoning: str = ""


# ============================================================================
# Final Output Models
# ============================================================================

class ScreeningOutput(BaseModel):
    """Final screening output - the main deliverable."""
    match_score: float = Field(ge=0.0, le=1.0, description="How well the candidate fits (0 = no fit, 1 = perfect fit)")
    recommendation: str = Field(description="Suggested next step")
    requires_human: bool = Field(description="Should a human double-check this decision?")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident is the system in its decision?")
    reasoning_summary: str = Field(description="Human-readable explanation of the decision")
    
    # Additional details for transparency
    skills_analysis: Optional[str] = None
    experience_analysis: Optional[str] = None
    flags: list[str] = Field(default_factory=list)


# ============================================================================
# Workflow State Model
# ============================================================================

class ScreeningState(BaseModel):
    """State that flows through the LangGraph workflow.
    
    Each agent reads from and writes to this shared state.
    """
    # Inputs
    resume_path: str = ""
    resume_raw_text: str = ""
    job_description: str = ""
    
    # Agent outputs (accumulated as workflow progresses)
    resume_data: Optional[ResumeData] = None
    extracted_skills: list[Skill] = Field(default_factory=list)
    job_requirements: Optional[JobRequirements] = None
    skills_match: Optional[SkillsMatchResult] = None
    experience_eval: Optional[ExperienceEvaluation] = None
    
    # Final output
    final_output: Optional[ScreeningOutput] = None
    
    # Workflow metadata
    errors: list[str] = Field(default_factory=list)
    agent_confidences: dict[str, float] = Field(default_factory=dict)
    workflow_complete: bool = False
    
    class Config:
        arbitrary_types_allowed = True
