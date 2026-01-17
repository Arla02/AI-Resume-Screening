"""Resume Parser Agent - Extracts structured information from raw resume text."""

from typing import Any

from .base import BaseAgent
from ..models import ResumeData, ContactInfo, Education, WorkExperience


class ResumeParserAgent(BaseAgent):
    """
    Agent responsible for parsing raw resume text into structured data.
    
    This agent takes messy, unstructured resume text and extracts:
    - Contact information
    - Education history
    - Work experience
    - Skills section
    - Certifications
    - Projects
    
    It handles various resume formats and messy data gracefully.
    """
    
    name = "ResumeParserAgent"
    description = "Parse raw resume text into structured sections (contact, education, experience, skills)"
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the resume and extract structured information.
        
        Args:
            state: Current workflow state containing resume_raw_text
            
        Returns:
            State updates with parsed resume_data
        """
        raw_text = state.get("resume_raw_text", "")
        
        if not raw_text:
            return {
                "resume_data": ResumeData(
                    parsing_confidence=0.0,
                    parsing_notes=["No resume text provided"]
                ),
                "errors": ["ResumeParserAgent: No resume text to parse"],
                "agent_confidences": {self.name: 0.0}
            }
        
        # Build the parsing prompt
        prompt = self._build_parsing_prompt(raw_text)
        
        # Call LLM
        response = await self._call_llm_async(prompt)
        
        # Parse the response
        resume_data = self._parse_response(response, raw_text)
        
        return {
            "resume_data": resume_data,
            "agent_confidences": {self.name: resume_data.parsing_confidence}
        }
    
    def _build_parsing_prompt(self, raw_text: str) -> str:
        """Build the prompt for resume parsing."""
        return f"""{self._build_system_prompt()}

TASK: Parse the following resume text into structured JSON format.

RESUME TEXT:
---
{raw_text[:8000]}  
---

Extract the following information and return as JSON:
{{
    "contact": {{
        "name": "Full name of the candidate",
        "email": "Email address",
        "phone": "Phone number",
        "location": "City, State/Country",
        "linkedin": "LinkedIn URL if present",
        "github": "GitHub URL if present"
    }},
    "summary": "Professional summary or objective if present",
    "education": [
        {{
            "degree": "Degree type (e.g., BS, MS, PhD)",
            "field": "Field of study",
            "institution": "School/University name",
            "graduation_year": "Year of graduation",
            "gpa": "GPA if mentioned"
        }}
    ],
    "work_experience": [
        {{
            "title": "Job title",
            "company": "Company name",
            "duration": "How long in this role",
            "start_date": "Start date",
            "end_date": "End date or 'Present'",
            "responsibilities": ["List of key responsibilities"],
            "technologies": ["Technologies/tools used in this role"]
        }}
    ],
    "skills_section": ["List of explicitly mentioned skills"],
    "certifications": ["List of certifications"],
    "projects": ["List of notable projects"],
    "parsing_confidence": 0.0 to 1.0 (how confident are you in this extraction),
    "parsing_notes": ["Any issues or uncertainties in parsing"]
}}

IMPORTANT:
- Extract what you can find, leave empty strings/arrays for missing info
- List work experience in reverse chronological order
- Include ALL skills mentioned anywhere in the resume
- Note any formatting issues or missing sections in parsing_notes
- Set parsing_confidence based on how complete and clear the resume was

Respond with ONLY valid JSON, no additional text."""
    
    def _parse_response(self, response: str, raw_text: str) -> ResumeData:
        """Parse the LLM response into a ResumeData object."""
        data = self._extract_json_from_response(response)
        
        if not data:
            # If JSON parsing fails, return a minimal result
            return ResumeData(
                raw_text=raw_text,
                parsing_confidence=0.3,
                parsing_notes=["Failed to parse LLM response as JSON"]
            )
        
        try:
            # Parse contact info
            contact_data = data.get("contact", {})
            contact = ContactInfo(
                name=contact_data.get("name", ""),
                email=contact_data.get("email", ""),
                phone=contact_data.get("phone", ""),
                location=contact_data.get("location", ""),
                linkedin=contact_data.get("linkedin", ""),
                github=contact_data.get("github", "")
            )
            
            # Parse education
            education = []
            for edu in data.get("education", []):
                education.append(Education(
                    degree=edu.get("degree", ""),
                    field=edu.get("field", ""),
                    institution=edu.get("institution", ""),
                    graduation_year=edu.get("graduation_year", ""),
                    gpa=edu.get("gpa", "")
                ))
            
            # Parse work experience
            work_experience = []
            for exp in data.get("work_experience", []):
                work_experience.append(WorkExperience(
                    title=exp.get("title", "Unknown"),
                    company=exp.get("company", "Unknown"),
                    duration=exp.get("duration", ""),
                    start_date=exp.get("start_date", ""),
                    end_date=exp.get("end_date", ""),
                    responsibilities=exp.get("responsibilities", []),
                    technologies=exp.get("technologies", [])
                ))
            
            return ResumeData(
                contact=contact,
                summary=data.get("summary", ""),
                education=education,
                work_experience=work_experience,
                skills_section=data.get("skills_section", []),
                certifications=data.get("certifications", []),
                projects=data.get("projects", []),
                raw_text=raw_text,
                parsing_confidence=float(data.get("parsing_confidence", 0.7)),
                parsing_notes=data.get("parsing_notes", [])
            )
        except Exception as e:
            return ResumeData(
                raw_text=raw_text,
                parsing_confidence=0.3,
                parsing_notes=[f"Error constructing ResumeData: {str(e)}"]
            )
