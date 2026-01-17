"""Decision Synthesizer Agent - Combines all inputs into final recommendation."""

from typing import Any

from .base import BaseAgent
from ..models import (
    ScreeningOutput, 
    SkillsMatchResult, 
    ExperienceEvaluation,
    ResumeData,
    JobRequirements
)
from ..config import get_config


class DecisionSynthesizerAgent(BaseAgent):
    """
    Agent responsible for making the final hiring recommendation.
    
    This agent:
    - Aggregates outputs from all other agents
    - Calculates final match score
    - Determines recommendation (Proceed/Reject/Manual Review)
    - Calculates overall confidence
    - Generates human-readable reasoning summary
    - Flags cases requiring human review
    """
    
    name = "DecisionSynthesizerAgent"
    description = "Synthesize all agent outputs into final recommendation with reasoning"
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Make the final screening decision.
        
        Args:
            state: Current workflow state with all agent outputs
            
        Returns:
            State updates with final_output and workflow_complete flag
        """
        # Gather all agent outputs
        resume_data = state.get("resume_data")
        job_requirements = state.get("job_requirements")
        skills_match = state.get("skills_match")
        experience_eval = state.get("experience_eval")
        agent_confidences = state.get("agent_confidences", {})
        errors = state.get("errors", [])
        
        # Convert to proper types if needed
        if isinstance(skills_match, dict):
            skills_match = SkillsMatchResult.model_validate(skills_match)
        if isinstance(experience_eval, dict):
            experience_eval = ExperienceEvaluation.model_validate(experience_eval)
        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)
        
        # Check for critical errors
        if errors or not skills_match or not experience_eval:
            return self._handle_error_case(state, errors)
        
        # Get config for thresholds
        config = get_config()
        
        # Calculate final scores
        match_score = self._calculate_match_score(skills_match, experience_eval)
        confidence = self._calculate_confidence(agent_confidences)
        requires_human = self._determine_human_review(
            match_score, confidence, errors, config
        )
        recommendation = self._determine_recommendation(
            match_score, requires_human, config
        )
        
        # Generate reasoning
        prompt = self._build_reasoning_prompt(
            resume_data, job_requirements, skills_match, experience_eval,
            match_score, recommendation, requires_human
        )
        reasoning = await self._call_llm_async(prompt)
        
        # Build final output
        final_output = ScreeningOutput(
            match_score=match_score,
            recommendation=recommendation,
            requires_human=requires_human,
            confidence=confidence,
            reasoning_summary=self._clean_reasoning(reasoning),
            skills_analysis=skills_match.reasoning if skills_match else None,
            experience_analysis=experience_eval.reasoning if experience_eval else None,
            flags=self._generate_flags(skills_match, experience_eval, errors)
        )
        
        return {
            "final_output": final_output,
            "workflow_complete": True,
            "agent_confidences": {self.name: confidence}
        }
    
    def _calculate_match_score(
        self, 
        skills_match: SkillsMatchResult, 
        experience_eval: ExperienceEvaluation
    ) -> float:
        """Calculate weighted final match score."""
        # Weights for different components
        SKILLS_WEIGHT = 0.6
        EXPERIENCE_WEIGHT = 0.4
        
        skills_score = skills_match.overall_score if skills_match else 0.0
        experience_score = experience_eval.experience_score if experience_eval else 0.0
        
        # Also factor in role relevance
        role_relevance = experience_eval.role_relevance if experience_eval else 0.5
        adjusted_exp_score = (experience_score * 0.7 + role_relevance * 0.3)
        
        final_score = (skills_score * SKILLS_WEIGHT) + (adjusted_exp_score * EXPERIENCE_WEIGHT)
        
        return round(min(max(final_score, 0.0), 1.0), 2)
    
    def _calculate_confidence(self, agent_confidences: dict[str, float]) -> float:
        """Calculate overall confidence from individual agent confidences."""
        if not agent_confidences:
            return 0.5
        
        # Use minimum confidence (weakest link)
        min_conf = min(agent_confidences.values())
        
        # Also consider average
        avg_conf = sum(agent_confidences.values()) / len(agent_confidences)
        
        # Weight toward minimum (be conservative)
        overall = (min_conf * 0.6) + (avg_conf * 0.4)
        
        return round(overall, 2)
    
    def _determine_human_review(
        self, 
        match_score: float, 
        confidence: float, 
        errors: list[str],
        config
    ) -> bool:
        """Determine if human review is required."""
        # Flag for human review if:
        # 1. Confidence is below threshold
        if confidence < config.confidence_threshold_low:
            return True
        
        # 2. Score is in ambiguous range
        if config.match_score_ambiguous_low <= match_score <= config.match_score_ambiguous_high:
            return True
        
        # 3. There were errors in processing
        if errors:
            return True
        
        return False
    
    def _determine_recommendation(
        self, 
        match_score: float, 
        requires_human: bool,
        config
    ) -> str:
        """Determine the recommendation based on score."""
        if requires_human and config.match_score_ambiguous_low <= match_score <= config.match_score_ambiguous_high:
            return "Needs manual review - borderline candidate"
        
        if match_score >= 0.75:
            return "Proceed to technical interview"
        elif match_score >= 0.6:
            return "Proceed to phone screening"
        elif match_score >= 0.4:
            return "Needs manual review"
        else:
            return "Reject - does not meet minimum requirements"
    
    def _build_reasoning_prompt(
        self,
        resume_data: ResumeData | None,
        job_requirements: JobRequirements | None,
        skills_match: SkillsMatchResult | None,
        experience_eval: ExperienceEvaluation | None,
        match_score: float,
        recommendation: str,
        requires_human: bool
    ) -> str:
        """Build prompt for generating reasoning summary."""
        candidate_name = resume_data.contact.name if resume_data and resume_data.contact else "The candidate"
        job_title = job_requirements.title if job_requirements else "the position"
        
        skills_summary = skills_match.reasoning if skills_match else "Skills analysis not available"
        exp_summary = experience_eval.reasoning if experience_eval else "Experience analysis not available"
        
        strengths = experience_eval.strengths if experience_eval else []
        gaps = experience_eval.gaps_identified if experience_eval else []
        
        return f"""Generate a concise, professional reasoning summary for a resume screening decision.

CONTEXT:
- Candidate: {candidate_name}
- Position: {job_title}
- Match Score: {match_score:.0%}
- Recommendation: {recommendation}
- Requires Human Review: {requires_human}

SKILLS ANALYSIS:
{skills_summary}

EXPERIENCE ANALYSIS:
{exp_summary}

STRENGTHS IDENTIFIED:
{chr(10).join('- ' + s for s in strengths) if strengths else '- None specifically identified'}

GAPS IDENTIFIED:
{chr(10).join('- ' + g for g in gaps) if gaps else '- None specifically identified'}

Write a 2-3 sentence summary that:
1. States the key reason for the recommendation
2. Mentions the most important strength or concern
3. If requires_human is true, explains why human review is suggested

Keep it professional, objective, and actionable. Do not include JSON or formatting, just plain text."""
    
    def _clean_reasoning(self, reasoning: str) -> str:
        """Clean up the reasoning text."""
        # Remove any JSON formatting if present
        if "```" in reasoning:
            # Extract text outside of code blocks
            parts = reasoning.split("```")
            reasoning = " ".join(parts[::2])  # Take every other part (outside code blocks)
        
        # Trim and clean
        reasoning = reasoning.strip()
        
        # Limit length
        if len(reasoning) > 500:
            reasoning = reasoning[:497] + "..."
        
        return reasoning
    
    def _generate_flags(
        self,
        skills_match: SkillsMatchResult | None,
        experience_eval: ExperienceEvaluation | None,
        errors: list[str]
    ) -> list[str]:
        """Generate flags for notable issues."""
        flags = []
        
        if errors:
            flags.append("Processing errors occurred")
        
        if skills_match:
            if skills_match.required_skills_met < skills_match.required_skills_total:
                missing = skills_match.required_skills_total - skills_match.required_skills_met
                flags.append(f"Missing {missing} required skill(s)")
        
        if experience_eval:
            if experience_eval.years_relevant < experience_eval.years_required:
                gap = experience_eval.years_required - experience_eval.years_relevant
                flags.append(f"Experience gap: {gap:.1f} years below requirement")
            
            if experience_eval.gaps_identified:
                flags.append(f"Experience gaps identified: {len(experience_eval.gaps_identified)}")
        
        return flags
    
    def _handle_error_case(self, state: dict, errors: list[str]) -> dict[str, Any]:
        """Handle cases where critical errors occurred."""
        error_summary = "; ".join(errors) if errors else "Unknown error"
        
        return {
            "final_output": ScreeningOutput(
                match_score=0.0,
                recommendation="Needs manual review - processing errors",
                requires_human=True,
                confidence=0.0,
                reasoning_summary=f"Could not complete automated screening due to errors: {error_summary}",
                flags=["Critical processing errors", "Manual review required"]
            ),
            "workflow_complete": True,
            "agent_confidences": {self.name: 0.0}
        }
