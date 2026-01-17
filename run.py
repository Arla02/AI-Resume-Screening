#!/usr/bin/env python3
"""
Agentic AI Resume Screening System

A command-line interface for screening resumes against job descriptions
using a multi-agent AI system.

Usage:
    python run.py --resume path/to/resume.pdf --job "Job description text"
    python run.py --resume path/to/resume.pdf --job-file path/to/job.txt
    python run.py --interactive
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Fix for Windows asyncio SSL cleanup errors on Python 3.10
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.workflow import create_screening_workflow
from src.models import ScreeningOutput


def print_colored(text: str, color: str = "white"):
    """Print colored text, with fallback if rich is not available."""
    if RICH_AVAILABLE:
        rprint(f"[{color}]{text}[/{color}]")
    else:
        print(text)


def print_result(result: ScreeningOutput):
    """Print the screening result in a formatted way."""
    if RICH_AVAILABLE:
        console = Console()
        
        # Create a table for the results
        table = Table(title="Resume Screening Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Determine color based on score
        score_color = "green" if result.match_score >= 0.7 else "yellow" if result.match_score >= 0.4 else "red"
        
        table.add_row("Match Score", f"[{score_color}]{result.match_score:.0%}[/{score_color}]")
        table.add_row("Recommendation", result.recommendation)
        table.add_row("Requires Human Review", "Yes" if result.requires_human else "No")
        table.add_row("Confidence", f"{result.confidence:.0%}")
        
        console.print(table)
        console.print()
        
        # Print reasoning
        console.print(Panel(result.reasoning_summary, title="Reasoning", border_style="blue"))
        
        # Print flags if any
        if result.flags:
            console.print()
            console.print("[yellow]Flags:[/yellow]")
            for flag in result.flags:
                console.print(f"  ⚠️  {flag}")
        
        # Print detailed analysis if available
        if result.skills_analysis:
            console.print()
            console.print(Panel(result.skills_analysis, title="Skills Analysis", border_style="dim"))
        
        if result.experience_analysis:
            console.print()
            console.print(Panel(result.experience_analysis, title="Experience Analysis", border_style="dim"))
    else:
        # Fallback plain text output
        print("\n" + "=" * 60)
        print("RESUME SCREENING RESULTS")
        print("=" * 60)
        print(f"Match Score:          {result.match_score:.0%}")
        print(f"Recommendation:       {result.recommendation}")
        print(f"Requires Human Review: {'Yes' if result.requires_human else 'No'}")
        print(f"Confidence:           {result.confidence:.0%}")
        print("-" * 60)
        print("REASONING:")
        print(result.reasoning_summary)
        if result.flags:
            print("-" * 60)
            print("FLAGS:")
            for flag in result.flags:
                print(f"  - {flag}")
        print("=" * 60)


def print_json(result: ScreeningOutput):
    """Print the result as JSON."""
    output = {
        "match_score": result.match_score,
        "recommendation": result.recommendation,
        "requires_human": result.requires_human,
        "confidence": result.confidence,
        "reasoning_summary": result.reasoning_summary,
    }
    if result.flags:
        output["flags"] = result.flags
    
    print(json.dumps(output, indent=2))


async def run_screening(
    resume_path: str,
    job_description: str,
    output_json: bool = False
) -> ScreeningOutput:
    """Run the screening workflow."""
    workflow = create_screening_workflow()
    
    if RICH_AVAILABLE and not output_json:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Screening resume...", total=None)
            result = await workflow.run(
                resume_path=resume_path,
                job_description=job_description
            )
            progress.remove_task(task)
    else:
        if not output_json:
            print("Screening resume...")
        result = await workflow.run(
            resume_path=resume_path,
            job_description=job_description
        )
    
    return result


def interactive_mode():
    """Run in interactive mode, prompting for inputs."""
    print_colored("\n🤖 Agentic AI Resume Screening System", "bold blue")
    print_colored("=" * 50, "dim")
    print()
    
    # Get resume path
    while True:
        resume_path = input("Enter path to resume (PDF, DOCX, or TXT): ").strip()
        if resume_path:
            path = Path(resume_path)
            if path.exists():
                break
            else:
                print_colored(f"File not found: {resume_path}", "red")
        else:
            print_colored("Please enter a valid path", "yellow")
    
    # Get job description
    print("\nEnter job description (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            if lines:
                break
            print("Please enter at least some job description text.")
    
    job_description = "\n".join(lines)
    
    # Run screening
    print()
    result = asyncio.run(run_screening(resume_path, job_description))
    
    # Print results
    print()
    print_result(result)
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Screen resumes against job descriptions using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --resume resume.pdf --job "We need a Python developer..."
  python run.py --resume resume.docx --job-file job_description.txt
  python run.py --interactive
  python run.py --resume resume.pdf --job "..." --json
        """
    )
    
    parser.add_argument(
        "--resume", "-r",
        help="Path to the resume file (PDF, DOCX, or TXT)"
    )
    parser.add_argument(
        "--job", "-j",
        help="Job description text"
    )
    parser.add_argument(
        "--job-file", "-jf",
        help="Path to a file containing the job description"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Validate inputs
    if not args.resume:
        parser.error("--resume is required (or use --interactive mode)")
    
    if not args.job and not args.job_file:
        parser.error("Either --job or --job-file is required")
    
    # Load job description
    if args.job_file:
        job_file_path = Path(args.job_file)
        if not job_file_path.exists():
            print_colored(f"Job description file not found: {args.job_file}", "red")
            sys.exit(1)
        job_description = job_file_path.read_text(encoding="utf-8")
    else:
        job_description = args.job
    
    # Check resume file exists
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print_colored(f"Resume file not found: {args.resume}", "red")
        sys.exit(1)
    
    # Run screening
    try:
        result = asyncio.run(run_screening(str(resume_path), job_description, args.json))
        
        if args.json:
            print_json(result)
        else:
            print()
            print_result(result)
            
    except KeyboardInterrupt:
        print("\nScreening cancelled.")
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error: {str(e)}", "red")
        if not args.json:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
