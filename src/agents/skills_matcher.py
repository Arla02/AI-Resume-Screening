"""Skills Matcher Agent - Compares candidate skills against job requirements."""

from typing import Any

from .base import BaseAgent
from ..models import Skill, JobRequirements, SkillMatch, SkillsMatchResult


class SkillsMatcherAgent(BaseAgent):
    """
    Agent responsible for matching candidate skills to job requirements.
    
    This agent:
    - Compares extracted skills against required/preferred skills
    - Handles semantic matching (e.g., "JS" = "JavaScript")
    - Scores each requirement match
    - Calculates overall skills match score
    """
    
    name = "SkillsMatcherAgent"
    description = "Compare candidate skills against job requirements and score the match"
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Match candidate skills against job requirements.
        
        Args:
            state: Current workflow state with extracted_skills and job_requirements
            
        Returns:
            State updates with skills_match result
        """
        extracted_skills = state.get("extracted_skills", [])
        job_requirements = state.get("job_requirements")
        
        if not job_requirements:
            return {
                "skills_match": SkillsMatchResult(confidence=0.0, reasoning="No job requirements to match against"),
                "errors": ["SkillsMatcherAgent: No job requirements available"],
                "agent_confidences": {self.name: 0.0}
            }
        
        # Convert to proper types if needed
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)
        
        skills_list = []
        for skill in extracted_skills:
            if isinstance(skill, dict):
                skills_list.append(Skill.model_validate(skill))
            else:
                skills_list.append(skill)
        
        # Build context for matching
        prompt = self._build_matching_prompt(skills_list, job_requirements)
        
        # Call LLM
        response = await self._call_llm_async(prompt)
        
        # Parse the response
        match_result = self._parse_response(response)
        
        return {
            "skills_match": match_result,
            "agent_confidences": {self.name: match_result.confidence}
        }
    
    def _build_matching_prompt(self, skills: list[Skill], requirements: JobRequirements) -> str:
        """Build the prompt for skills matching."""
        # Format candidate skills
        skills_text = "\n".join([
            f"- {s.name} ({s.category}, {s.proficiency})"
            for s in skills
        ])
        
        # Format job requirements
        required_text = ", ".join(requirements.required_skills) if requirements.required_skills else "None specified"
        preferred_text = ", ".join(requirements.preferred_skills) if requirements.preferred_skills else "None specified"
        
        return f"""{self._build_system_prompt()}

TASK: Match the candidate's skills against the job requirements.

CANDIDATE SKILLS:
{skills_text if skills_text else "No skills extracted"}

JOB REQUIREMENTS:
- Required Skills: {required_text}
- Preferred Skills: {preferred_text}
- Min Experience: {requirements.min_years_experience} years

For each required and preferred skill, determine if the candidate has it.
Consider semantic matches (e.g., "JavaScript" matches "JS", "React" matches "ReactJS").
Consider related skills (e.g., "Python" partially matches "programming experience").

Return as JSON:
{{
    "matches": [
        {{
            "requirement": "The skill requirement from job description",
            "matched": true/false,
            "matched_skill": "The candidate skill that matches (if any)",
            "match_quality": "exact|semantic|partial|none",
            "confidence": 0.0-1.0,
            "notes": "Any relevant notes"
        }}
    ],
    "required_skills_met": number of required skills the candidate has,
    "required_skills_total": total number of required skills,
    "preferred_skills_met": number of preferred skills the candidate has,
    "preferred_skills_total": total number of preferred skills,
    "overall_score": 0.0-1.0 (weighted score: required skills count more than preferred),
    "confidence": 0.0-1.0 (how confident are you in this matching),
    "reasoning": "Summary of the skills match analysis"
}}

SCORING GUIDELINES:
- Required skills should account for ~70% of the overall_score
- Preferred skills should account for ~30% of the overall_score
- Exact matches = full credit
- Semantic matches = 90% credit
- Partial matches = 50% credit
- No match = 0% credit

Respond with ONLY valid JSON."""
    
    def _parse_response(self, response: str) -> SkillsMatchResult:
        """Parse the LLM response into SkillsMatchResult."""
        data = self._extract_json_from_response(response)
        
        if not data:
            return SkillsMatchResult(
                confidence=0.3,
                reasoning="Failed to parse matching results"
            )
        
        try:
            matches = []
            for match_data in data.get("matches", []):
                match = SkillMatch(
                    requirement=match_data.get("requirement", "") or "",
                    matched=match_data.get("matched", False),
                    matched_skill=match_data.get("matched_skill") or "",  # Handle null
                    match_quality=match_data.get("match_quality", "none") or "none",
                    confidence=float(match_data.get("confidence", 0.5) or 0.5),
                    notes=match_data.get("notes", "") or ""
                )
                matches.append(match)
            
            return SkillsMatchResult(
                matches=matches,
                required_skills_met=int(data.get("required_skills_met", 0)),
                required_skills_total=int(data.get("required_skills_total", 0)),
                preferred_skills_met=int(data.get("preferred_skills_met", 0)),
                preferred_skills_total=int(data.get("preferred_skills_total", 0)),
                overall_score=float(data.get("overall_score", 0.0)),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", "")
            )
        except Exception as e:
            print(f"[{self.name}] Error parsing match result: {e}")
            return SkillsMatchResult(
                confidence=0.3,
                reasoning=f"Error parsing results: {str(e)}"
            )
