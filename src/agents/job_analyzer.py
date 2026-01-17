"""Job Analyzer Agent - Parses job description into structured requirements."""

from typing import Any

from .base import BaseAgent
from ..models import JobRequirements, Requirement


class JobAnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing job descriptions.
    
    This agent:
    - Extracts required vs preferred qualifications
    - Identifies must-have skills and nice-to-haves
    - Determines experience requirements
    - Parses education and certification requirements
    """
    
    name = "JobAnalyzerAgent"
    description = "Parse job descriptions into structured requirements (required skills, experience, education)"
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze the job description and extract requirements.
        
        Args:
            state: Current workflow state containing job_description
            
        Returns:
            State updates with job_requirements
        """
        job_description = state.get("job_description", "")
        
        if not job_description:
            return {
                "job_requirements": JobRequirements(parsing_confidence=0.0),
                "errors": ["JobAnalyzerAgent: No job description provided"],
                "agent_confidences": {self.name: 0.0}
            }
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(job_description)
        
        # Call LLM
        response = await self._call_llm_async(prompt)
        
        # Parse the response
        job_requirements = self._parse_response(response)
        
        return {
            "job_requirements": job_requirements,
            "agent_confidences": {self.name: job_requirements.parsing_confidence}
        }
    
    def _build_analysis_prompt(self, job_description: str) -> str:
        """Build the prompt for job description analysis."""
        return f"""{self._build_system_prompt()}

TASK: Analyze the following job description and extract structured requirements.

JOB DESCRIPTION:
---
{job_description[:6000]}
---

Extract the following information and return as JSON:
{{
    "title": "Job title",
    "summary": "Brief summary of the role",
    "required_skills": ["List of MUST-HAVE skills"],
    "preferred_skills": ["List of NICE-TO-HAVE skills"],
    "min_years_experience": 0 (minimum years of relevant experience required, 0 if not specified),
    "education_requirements": ["Required degrees or educational background"],
    "certifications_required": ["Any required certifications"],
    "responsibilities": ["Key job responsibilities"],
    "requirements": [
        {{
            "description": "Specific requirement description",
            "category": "skill|experience|education|certification|other",
            "priority": "required|preferred|nice_to_have",
            "years_needed": null or number
        }}
    ],
    "parsing_confidence": 0.0 to 1.0
}}

GUIDELINES:
- Distinguish between REQUIRED (must-have) and PREFERRED (nice-to-have) qualifications
- Extract specific years of experience if mentioned (e.g., "3+ years" -> 3)
- Include both technical and soft skill requirements
- If something says "preferred" or "nice to have", put in preferred_skills
- If something says "required" or "must have", put in required_skills
- When unclear, assume requirements are preferred rather than required
- Include specific technologies, tools, and frameworks mentioned

Respond with ONLY valid JSON."""
    
    def _parse_response(self, response: str) -> JobRequirements:
        """Parse the LLM response into JobRequirements object."""
        data = self._extract_json_from_response(response)
        
        if not data:
            return JobRequirements(parsing_confidence=0.3)
        
        try:
            # Parse structured requirements
            requirements = []
            for req_data in data.get("requirements", []):
                req = Requirement(
                    description=req_data.get("description", ""),
                    category=req_data.get("category", "other"),
                    priority=req_data.get("priority", "required"),
                    years_needed=req_data.get("years_needed")
                )
                requirements.append(req)
            
            return JobRequirements(
                title=data.get("title", ""),
                summary=data.get("summary", ""),
                required_skills=data.get("required_skills", []),
                preferred_skills=data.get("preferred_skills", []),
                min_years_experience=int(data.get("min_years_experience", 0)),
                education_requirements=data.get("education_requirements", []),
                certifications_required=data.get("certifications_required", []),
                responsibilities=data.get("responsibilities", []),
                requirements=requirements,
                parsing_confidence=float(data.get("parsing_confidence", 0.7))
            )
        except Exception as e:
            print(f"[{self.name}] Error parsing job requirements: {e}")
            return JobRequirements(parsing_confidence=0.3)
