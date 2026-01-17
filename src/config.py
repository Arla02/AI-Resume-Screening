"""Configuration management for the resume screening system."""

import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    
    # LLM Provider: "gemini" or "groq"
    llm_provider: Literal["gemini", "groq"] = "gemini"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    
    # Groq Configuration
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # Fast and capable
    
    # Common settings
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # Confidence Thresholds
    confidence_threshold_low: float = 0.6
    match_score_ambiguous_low: float = 0.4
    match_score_ambiguous_high: float = 0.7
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        # Determine provider
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        # Get API keys
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        
        # Auto-detect provider if not specified
        if provider == "gemini" and not gemini_key and groq_key:
            provider = "groq"
        elif provider == "groq" and not groq_key and gemini_key:
            provider = "gemini"
        
        # Validate
        if provider == "gemini" and (not gemini_key or gemini_key == "your_api_key_here"):
            if groq_key and groq_key != "your_groq_api_key_here":
                provider = "groq"  # Fallback to Groq if available
            else:
                raise ValueError(
                    "No API key configured. Please set GEMINI_API_KEY or GROQ_API_KEY in your .env file.\n"
                    "Get a free Gemini key at: https://makersuite.google.com/app/apikey\n"
                    "Get a free Groq key at: https://console.groq.com"
                )
        
        if provider == "groq" and (not groq_key or groq_key == "your_groq_api_key_here"):
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        
        return cls(
            llm_provider=provider,
            gemini_api_key=gemini_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            groq_api_key=groq_key,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            confidence_threshold_low=float(os.getenv("CONFIDENCE_THRESHOLD_LOW", "0.6")),
            match_score_ambiguous_low=float(os.getenv("MATCH_SCORE_AMBIGUOUS_LOW", "0.4")),
            match_score_ambiguous_high=float(os.getenv("MATCH_SCORE_AMBIGUOUS_HIGH", "0.7")),
        )


# Global config instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None
