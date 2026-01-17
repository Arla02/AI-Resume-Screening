"""LangGraph workflow for orchestrating the resume screening agents."""

import asyncio
from typing import Any, Annotated, TypedDict

from langgraph.graph import StateGraph, END

from .agents import (
    ResumeParserAgent,
    SkillExtractorAgent,
    JobAnalyzerAgent,
    SkillsMatcherAgent,
    ExperienceEvaluatorAgent,
    DecisionSynthesizerAgent,
)
from .models import ScreeningState, ScreeningOutput
from .document_parser import parse_document


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    """Reducer function to merge dictionaries from parallel branches."""
    result = dict(a) if a else {}
    if b:
        result.update(b)
    return result


def merge_lists(a: list | None, b: list | None) -> list:
    """Reducer function to merge lists from parallel branches."""
    result = list(a) if a else []
    if b:
        for item in b:
            if item not in result:
                result.append(item)
    return result


class WorkflowState(TypedDict, total=False):
    """State dictionary for the LangGraph workflow with reducers for parallel updates."""
    # Inputs
    resume_path: str
    resume_raw_text: str
    job_description: str
    
    # Agent outputs
    resume_data: dict | None
    extracted_skills: list
    job_requirements: dict | None
    skills_match: dict | None
    experience_eval: dict | None
    
    # Final output
    final_output: dict | None
    
    # Metadata - these use Annotated with reducers for parallel updates
    errors: Annotated[list, merge_lists]
    agent_confidences: Annotated[dict, merge_dicts]
    workflow_complete: bool


class ResumeScreeningWorkflow:
    """
    LangGraph workflow for agentic resume screening.
    
    This workflow orchestrates multiple agents to:
    1. Parse the resume document
    2. Extract and structure resume information
    3. Analyze job requirements
    4. Match skills and evaluate experience
    5. Synthesize a final recommendation
    
    The workflow uses parallel branches where possible for efficiency,
    and converges results for final decision making.
    """
    
    def __init__(self):
        """Initialize the workflow with all agents."""
        self.resume_parser = ResumeParserAgent()
        self.skill_extractor = SkillExtractorAgent()
        self.job_analyzer = JobAnalyzerAgent()
        self.skills_matcher = SkillsMatcherAgent()
        self.experience_evaluator = ExperienceEvaluatorAgent()
        self.decision_synthesizer = DecisionSynthesizerAgent()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        
        # Create the graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (each node is an agent's process function)
        workflow.add_node("parse_document", self._parse_document_node)
        workflow.add_node("parse_resume", self._parse_resume_node)
        workflow.add_node("analyze_job", self._analyze_job_node)
        workflow.add_node("extract_skills", self._extract_skills_node)
        workflow.add_node("match_skills", self._match_skills_node)
        workflow.add_node("evaluate_experience", self._evaluate_experience_node)
        workflow.add_node("synthesize_decision", self._synthesize_decision_node)
        
        # Define the workflow edges
        # Entry point: parse the document
        workflow.set_entry_point("parse_document")
        
        # After document parsing, parse resume and analyze job in parallel
        workflow.add_edge("parse_document", "parse_resume")
        workflow.add_edge("parse_document", "analyze_job")
        
        # After resume parsing, extract skills
        workflow.add_edge("parse_resume", "extract_skills")
        
        # Skills matching depends on both skill extraction and job analysis
        workflow.add_edge("extract_skills", "match_skills")
        workflow.add_edge("analyze_job", "match_skills")
        
        # Experience evaluation depends on resume parsing and job analysis
        workflow.add_edge("parse_resume", "evaluate_experience")
        workflow.add_edge("analyze_job", "evaluate_experience")
        
        # Decision synthesis depends on skills matching and experience evaluation
        workflow.add_edge("match_skills", "synthesize_decision")
        workflow.add_edge("evaluate_experience", "synthesize_decision")
        
        # End after decision synthesis
        workflow.add_edge("synthesize_decision", END)
        
        return workflow.compile()
    
    def _parse_document_node(self, state: WorkflowState) -> dict:
        """Node: Parse the resume document to extract raw text."""
        resume_path = state.get("resume_path", "")
        
        if not resume_path:
            # If raw text is already provided, skip parsing
            if state.get("resume_raw_text"):
                return {}
            return {
                "errors": ["No resume path or text provided"]
            }
        
        # Parse the document
        result = parse_document(resume_path)
        
        if not result.success:
            return {
                "errors": [f"Document parsing failed: {result.error_message}"],
                "resume_raw_text": ""
            }
        
        return {
            "resume_raw_text": result.text,
            "agent_confidences": {"DocumentParser": result.confidence}
        }
    
    async def _parse_resume_node(self, state: WorkflowState) -> dict:
        """Node: Parse resume text into structured data."""
        return await self.resume_parser.process(dict(state))
    
    async def _analyze_job_node(self, state: WorkflowState) -> dict:
        """Node: Analyze job description."""
        return await self.job_analyzer.process(dict(state))
    
    async def _extract_skills_node(self, state: WorkflowState) -> dict:
        """Node: Extract skills from parsed resume."""
        return await self.skill_extractor.process(dict(state))
    
    async def _match_skills_node(self, state: WorkflowState) -> dict:
        """Node: Match skills against requirements."""
        return await self.skills_matcher.process(dict(state))
    
    async def _evaluate_experience_node(self, state: WorkflowState) -> dict:
        """Node: Evaluate work experience."""
        return await self.experience_evaluator.process(dict(state))
    
    async def _synthesize_decision_node(self, state: WorkflowState) -> dict:
        """Node: Synthesize final decision."""
        return await self.decision_synthesizer.process(dict(state))
    
    async def run(
        self,
        resume_path: str = "",
        resume_text: str = "",
        job_description: str = ""
    ) -> ScreeningOutput:
        """
        Run the complete screening workflow.
        
        Args:
            resume_path: Path to resume file (PDF, DOCX, or TXT)
            resume_text: Raw resume text (alternative to file path)
            job_description: The job description text
            
        Returns:
            ScreeningOutput with match score, recommendation, and reasoning
        """
        # Initialize state
        initial_state: WorkflowState = {
            "resume_path": resume_path,
            "resume_raw_text": resume_text,
            "job_description": job_description,
            "resume_data": None,
            "extracted_skills": [],
            "job_requirements": None,
            "skills_match": None,
            "experience_eval": None,
            "final_output": None,
            "errors": [],
            "agent_confidences": {},
            "workflow_complete": False,
        }
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Extract and return the final output
        output_data = final_state.get("final_output")
        
        if output_data:
            if isinstance(output_data, dict):
                return ScreeningOutput.model_validate(output_data)
            return output_data
        
        # Fallback if no output
        return ScreeningOutput(
            match_score=0.0,
            recommendation="Error - workflow did not complete",
            requires_human=True,
            confidence=0.0,
            reasoning_summary="The workflow failed to produce a result. Please review manually.",
            flags=["Workflow error"]
        )
    
    def run_sync(
        self,
        resume_path: str = "",
        resume_text: str = "",
        job_description: str = ""
    ) -> ScreeningOutput:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(resume_path, resume_text, job_description))


def create_screening_workflow() -> ResumeScreeningWorkflow:
    """Factory function to create a screening workflow."""
    return ResumeScreeningWorkflow()


async def screen_resume(
    resume_path: str = "",
    resume_text: str = "",
    job_description: str = ""
) -> ScreeningOutput:
    """
    Convenience function to screen a resume.
    
    Args:
        resume_path: Path to resume file
        resume_text: Raw resume text (alternative to path)
        job_description: Job description text
        
    Returns:
        ScreeningOutput with recommendation and reasoning
    """
    workflow = create_screening_workflow()
    return await workflow.run(resume_path, resume_text, job_description)
