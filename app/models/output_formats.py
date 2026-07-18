"""
Structured output format definitions.

Each OutputFormat maps to a Pydantic model describing the exact JSON
shape that format should produce. These models serve double duty:
  1. They're what OutputParser.validate_output() validates parsed JSON
     against (app/services/output_parser.py) -- if the LLM's JSON is
     missing a field or has the wrong type, Pydantic catches it here.
  2. Their field names/types are the source of truth for the format
     instruction text the LLM is told to follow (see
     OutputParser.build_format_instruction), so the instruction and the
     validation can never drift out of sync from each other by accident
     as long as both are kept in view of this file.
"""

from enum import Enum

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    """
    What shape the caller wants ChatResponse.response to be parseable
    into. TEXT (the default) means "just give me free-form text, don't
    parse anything" -- every other value triggers OutputParser.
    """

    TEXT = "text"
    SENTIMENT = "sentiment"
    SUMMARY = "summary"
    CODE_REVIEW = "code_review"
    QA = "qa"
    JSON = "json"  # caller-defined shape, described via schema_hint


class SentimentAnalysis(BaseModel):
    sentiment: str  # "positive", "negative", "neutral", "mixed"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Summary(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    word_count: int


class CodeReview(BaseModel):
    language: str
    issues: list[str]
    suggestions: list[str]
    quality_score: int = Field(ge=1, le=10)
    explanation: str


class QuestionAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_or_reasoning: str
    follow_up_questions: list[str] = Field(default_factory=list)


class CustomFormat(BaseModel):
    """
    For OutputFormat.JSON, where the caller describes their own desired
    shape via schema_hint instead of picking one of our named formats.
    We can't validate field-by-field against something we don't have a
    model for, so this just requires "is a JSON object at all" --
    `data: dict` accepts any valid JSON object.
    """

    data: dict
