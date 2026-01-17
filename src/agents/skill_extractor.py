"""Skill Extractor Agent - Identifies and categorizes candidate skills."""

from typing import Any

from .base import BaseAgent
from ..models import Skill, ResumeData


class SkillExtractorAgent(BaseAgent):
    """
    Agent responsible for extracting and categorizing skills from parsed resume.
    
    This agent:
    - Identifies explicit skills (from skills section)
    - Infers implicit skills (from job responsibilities and projects)
    - Categorizes skills (technical, soft, tools, languages, frameworks)
    - Estimates proficiency level
    """
    
    name = "SkillExtractorAgent"
    description = "Extract and categorize technical and soft skills from resume data"
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Extract skills from the parsed resume.
        
        Args:
            state: Current workflow state containing resume_data
            
        Returns:
            State updates with extracted_skills list
        """
        resume_data = state.get("resume_data")
        
        if not resume_data:
            return {
                "extracted_skills": [],
                "errors": ["SkillExtractorAgent: No resume data available"],
                "agent_confidences": {self.name: 0.0}
            }
        
        # If resume_data is a dict, convert to ResumeData
        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)
        
        # Build context from resume
        context = self._build_context(resume_data)
        
        # Build the extraction prompt
        prompt = self._build_extraction_prompt(context)
        
        # Call LLM
        response = await self._call_llm_async(prompt)
        
        # Parse the response
        skills, confidence = self._parse_response(response)
        
        return {
            "extracted_skills": skills,
            "agent_confidences": {self.name: confidence}
        }
    
    def _build_context(self, resume_data: ResumeData) -> str:
        """Build context string from resume data for skill extraction."""
        parts = []
        
        # Skills section
        if resume_data.skills_section:
            parts.append("EXPLICIT SKILLS SECTION:")
            parts.append(", ".join(resume_data.skills_section))
            parts.append("")
        
        # Work experience (for inferring skills)
        if resume_data.work_experience:
            parts.append("WORK EXPERIENCE:")
            for exp in resume_data.work_experience:
                parts.append(f"\n{exp.title} at {exp.company}")
                if exp.responsibilities:
                    parts.append("Responsibilities: " + "; ".join(exp.responsibilities))
                if exp.technologies:
                    parts.append("Technologies: " + ", ".join(exp.technologies))
            parts.append("")
        
        # Projects
        if resume_data.projects:
            parts.append("PROJECTS:")
            parts.append("; ".join(resume_data.projects))
            parts.append("")
        
        # Certifications
        if resume_data.certifications:
            parts.append("CERTIFICATIONS:")
            parts.append(", ".join(resume_data.certifications))
        
        return "\n".join(parts)
    
    def _build_extraction_prompt(self, context: str) -> str:
        """Build the prompt for skill extraction."""
        return f"""{self._build_system_prompt()}

TASK: Extract and categorize ALL skills from the following resume information.

RESUME INFORMATION:
---
{context[:6000]}
---

For each skill, determine:
1. The skill name (use standard/common names when possible)
2. Category: technical, soft_skill, tool, language, framework, or other
3. Proficiency: beginner, intermediate, advanced, or expert
4. Source: explicit (directly listed) or inferred (from context)
5. Confidence: 0.0-1.0 for how certain you are this is a real skill

Return as JSON:
{{
    "skills": [
        {{
            "name": "Python",
            "category": "language",
            "proficiency": "advanced",
            "source": "explicit",
            "confidence": 0.95
        }},
        {{
            "name": "Team Leadership",
            "category": "soft_skill",
            "proficiency": "intermediate",
            "source": "inferred",
            "confidence": 0.7
        }}
    ],
    "extraction_confidence": 0.0 to 1.0 (overall confidence in extraction),
    "notes": "Any observations about the skill profile"
}}

GUIDELINES:
- Include BOTH technical and soft skills
- Normalize skill names (e.g., "JS" -> "JavaScript", "ML" -> "Machine Learning")
- Infer skills from job responsibilities (e.g., "led team of 5" implies leadership)
- Consider certifications as evidence of skills
- Don't duplicate skills - if Python appears multiple times, list once with highest proficiency
- Be conservative with proficiency estimates unless there's clear evidence

Respond with ONLY valid JSON."""
    
    def _parse_response(self, response: str) -> tuple[list[Skill], float]:
        """Parse the LLM response into skill objects."""
        data = self._extract_json_from_response(response)
        
        if not data:
            return [], 0.3
        
        # Valid categories
        valid_categories = {"technical", "soft_skill", "tool", "language", "framework", "other"}
        valid_proficiencies = {"beginner", "intermediate", "advanced", "expert"}
        valid_sources = {"explicit", "inferred"}
        
        skills = []
        try:
            for skill_data in data.get("skills", []):
                # Validate and fix category
                category = skill_data.get("category", "other")
                if category not in valid_categories:
                    category = "other"  # Map invalid categories to "other"
                
                # Validate proficiency
                proficiency = skill_data.get("proficiency", "intermediate")
                if proficiency not in valid_proficiencies:
                    proficiency = "intermediate"
                
                # Validate source
                source = skill_data.get("source", "explicit")
                if source not in valid_sources:
                    source = "explicit"
                
                skill = Skill(
                    name=skill_data.get("name", "Unknown") or "Unknown",
                    category=category,
                    proficiency=proficiency,
                    confidence=float(skill_data.get("confidence", 0.7) or 0.7),
                    source=source
                )
                skills.append(skill)
            
            confidence = float(data.get("extraction_confidence", 0.7) or 0.7)
            return skills, confidence
        except Exception as e:
            print(f"[{self.name}] Error parsing skills: {e}")
            return [], 0.3

