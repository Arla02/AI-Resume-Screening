"""Experience Evaluator Agent - Assesses work experience relevance."""

from typing import Any

from .base import BaseAgent
from ..models import ResumeData, JobRequirements, ExperienceEvaluation


class ExperienceEvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating work experience.
    
    This agent:
    - Calculates total relevant experience
    - Assesses role relevance to target position
    - Evaluates career progression
    - Identifies experience gaps
    """
    
    name = "ExperienceEvaluatorAgent"
    description = "Evaluate work experience relevance, career progression, and identify gaps"
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate the candidate's work experience.
        
        Args:
            state: Current workflow state with resume_data and job_requirements
            
        Returns:
            State updates with experience_eval
        """
        resume_data = state.get("resume_data")
        job_requirements = state.get("job_requirements")
        
        if not resume_data or not job_requirements:
            return {
                "experience_eval": ExperienceEvaluation(
                    confidence=0.0,
                    reasoning="Missing resume or job requirements data"
                ),
                "errors": ["ExperienceEvaluatorAgent: Missing required data"],
                "agent_confidences": {self.name: 0.0}
            }
        
        # Convert to proper types if needed
        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(resume_data, job_requirements)
        
        # Call LLM
        response = await self._call_llm_async(prompt)
        
        # Parse the response
        evaluation = self._parse_response(response)
        
        return {
            "experience_eval": evaluation,
            "agent_confidences": {self.name: evaluation.confidence}
        }
    
    def _build_evaluation_prompt(self, resume: ResumeData, requirements: JobRequirements) -> str:
        """Build the prompt for experience evaluation."""
        # Format work experience
        experience_text = ""
        for i, exp in enumerate(resume.work_experience, 1):
            experience_text += f"""
{i}. {exp.title} at {exp.company}
   Duration: {exp.duration or f'{exp.start_date} - {exp.end_date}'}
   Responsibilities: {'; '.join(exp.responsibilities[:5]) if exp.responsibilities else 'Not specified'}
   Technologies: {', '.join(exp.technologies) if exp.technologies else 'Not specified'}
"""
        
        return f"""{self._build_system_prompt()}

TASK: Evaluate the candidate's work experience against the job requirements.

CANDIDATE'S WORK EXPERIENCE:
{experience_text if experience_text else "No work experience listed"}

EDUCATION:
{', '.join([f"{e.degree} in {e.field} from {e.institution}" for e in resume.education]) if resume.education else "Not specified"}

JOB REQUIREMENTS:
- Title: {requirements.title}
- Min Years Experience: {requirements.min_years_experience}
- Key Responsibilities: {', '.join(requirements.responsibilities[:5]) if requirements.responsibilities else 'Not specified'}
- Required Skills: {', '.join(requirements.required_skills[:10]) if requirements.required_skills else 'Not specified'}

Evaluate and return as JSON:
{{
    "years_relevant": estimated years of RELEVANT experience (not just total years),
    "years_required": {requirements.min_years_experience},
    "experience_score": 0.0-1.0 (does experience meet requirements?),
    "role_relevance": 0.0-1.0 (how relevant are past roles to this position?),
    "career_progression": "Description of career trajectory (e.g., 'steady growth', 'lateral moves', 'career change')",
    "gaps_identified": ["List of experience gaps or concerns"],
    "strengths": ["List of experience strengths"],
    "confidence": 0.0-1.0,
    "reasoning": "Summary of experience evaluation"
}}

EVALUATION CRITERIA:
1. Years of Experience:
   - Compare relevant experience to minimum required
   - Weight recent experience more heavily
   - Consider internships as partial experience

2. Role Relevance:
   - How similar are past job titles to target role?
   - How transferable are past responsibilities?
   - Industry relevance

3. Career Progression:
   - Is there growth in responsibilities?
   - Logical career path toward this role?
   - Any concerning patterns (frequent job changes, long gaps)?

4. Experience Gaps:
   - Missing experience in key areas
   - Lack of leadership experience if required
   - Never worked at scale if role requires it

Respond with ONLY valid JSON."""
    
    def _parse_response(self, response: str) -> ExperienceEvaluation:
        """Parse the LLM response into ExperienceEvaluation."""
        data = self._extract_json_from_response(response)
        
        if not data:
            return ExperienceEvaluation(
                confidence=0.3,
                reasoning="Failed to parse evaluation results"
            )
        
        try:
            return ExperienceEvaluation(
                years_relevant=float(data.get("years_relevant", 0)),
                years_required=int(data.get("years_required", 0)),
                experience_score=float(data.get("experience_score", 0.5)),
                role_relevance=float(data.get("role_relevance", 0.5)),
                career_progression=data.get("career_progression", ""),
                gaps_identified=data.get("gaps_identified", []),
                strengths=data.get("strengths", []),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", "")
            )
        except Exception as e:
            print(f"[{self.name}] Error parsing evaluation: {e}")
            return ExperienceEvaluation(
                confidence=0.3,
                reasoning=f"Error parsing results: {str(e)}"
            )
