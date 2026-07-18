"""
Output parsing service -- turns an LLM's free-form text reply into
validated structured data.

LLMs are trained to write for humans, not machines: even when explicitly
told "respond only with JSON", they'll often wrap it in a markdown code
fence, add a friendly sentence before or after it, or otherwise decorate
what should be a clean machine-readable payload. This module's job is to
absorb that unpredictability so the rest of the app only ever deals with
validated Pydantic objects, never raw LLM text, once structured output
is requested.

The flow (see parse(), the main entry point):
  1. build_format_instruction() tells the model exactly what shape to
     produce, appended to the user's prompt before it's ever sent.
  2. extract_json() digs a JSON object out of whatever text comes back,
     trying progressively looser strategies.
  3. validate_output() checks that JSON against the Pydantic model for
     the requested format -- catching wrong types/missing fields that
     valid-JSON-but-wrong-shape output would otherwise let through.
  4. If either step fails, build_retry_prompt() constructs a stricter
     follow-up prompt so the router (app/routers/chat.py) can ask the
     model to try again.
"""

import json
import re

import structlog
from pydantic import BaseModel, ValidationError

from app.core.exceptions import OutputParsingError
from app.models.output_formats import (
    CodeReview,
    CustomFormat,
    OutputFormat,
    QuestionAnswer,
    SentimentAnalysis,
    Summary,
)

logger = structlog.get_logger()

# Which Pydantic model validates which format's JSON. TEXT is
# deliberately absent -- parse() short-circuits before ever consulting
# this map for TEXT, since "no parsing" has no model to validate against.
FORMAT_MODEL_MAP: dict[OutputFormat, type[BaseModel]] = {
    OutputFormat.SENTIMENT: SentimentAnalysis,
    OutputFormat.SUMMARY: Summary,
    OutputFormat.CODE_REVIEW: CodeReview,
    OutputFormat.QA: QuestionAnswer,
    OutputFormat.JSON: CustomFormat,
}

_FENCED_BLOCK_PATTERNS = (
    r"```json\s*(.*?)\s*```",
    r"```\s*(.*?)\s*```",
)


class OutputParser:
    """Stateless -- same reasoning as PromptService: no per-instance state,
    a single shared instance is used purely for consistency with the
    other services (see app/core/dependencies.py)."""

    def build_format_instruction(
        self, output_format: OutputFormat, schema_hint: str | None
    ) -> str:
        """
        The text appended to the user's prompt telling the model exactly
        what JSON shape to emit. Kept in lockstep with the field names on
        each format's Pydantic model in output_formats.py -- if a field
        gets renamed there, this instruction needs to change too.
        """
        if output_format == OutputFormat.TEXT:
            return ""

        if output_format == OutputFormat.SENTIMENT:
            return (
                "Respond ONLY with a JSON object in this exact format, "
                "no other text:\n"
                "{\n"
                '    "sentiment": "positive|negative|neutral|mixed",\n'
                '    "confidence": 0.0 to 1.0,\n'
                '    "reasoning": "brief explanation"\n'
                "}"
            )

        if output_format == OutputFormat.SUMMARY:
            return (
                "Respond ONLY with a JSON object in this exact format, "
                "no other text:\n"
                "{\n"
                '    "title": "short title",\n'
                '    "summary": "concise summary",\n'
                '    "key_points": ["point 1", "point 2", "point 3"],\n'
                '    "word_count": number\n'
                "}"
            )

        if output_format == OutputFormat.CODE_REVIEW:
            return (
                "Respond ONLY with a JSON object in this exact format, "
                "no other text:\n"
                "{\n"
                '    "language": "programming language",\n'
                '    "issues": ["issue 1", "issue 2"],\n'
                '    "suggestions": ["suggestion 1", "suggestion 2"],\n'
                '    "quality_score": 1 to 10,\n'
                '    "explanation": "overall assessment"\n'
                "}"
            )

        if output_format == OutputFormat.QA:
            return (
                "Respond ONLY with a JSON object in this exact format, "
                "no other text:\n"
                "{\n"
                '    "answer": "your answer",\n'
                '    "confidence": 0.0 to 1.0,\n'
                '    "sources_or_reasoning": "how you arrived at this answer",\n'
                '    "follow_up_questions": ["question 1", "question 2"]\n'
                "}"
            )

        # OutputFormat.JSON -- caller-defined shape.
        if schema_hint:
            return f"Respond ONLY with a JSON object matching this schema: {schema_hint}"
        return "Respond ONLY with a valid JSON object, no other text."

    def extract_json(self, raw_text: str) -> dict:
        """
        Tries progressively looser strategies to find a JSON object
        inside arbitrary LLM output, since "the model did exactly what it
        was told" is the exception, not the rule, for small local models.
        """
        text = raw_text.strip()

        # Attempt 1: the clean case -- the whole response IS the JSON.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: JSON inside a markdown code fence (```json ... ```
        # or a plain ``` ... ``` block), which models love to add even
        # when told not to.
        for pattern in _FENCED_BLOCK_PATTERNS:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Attempt 3: JSON buried in prose -- slice from the first { to
        # the last } and hope everything in between is valid. This is
        # the loosest strategy (a stray brace elsewhere in the prose
        # would break it), so it's the last resort.
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            try:
                return json.loads(text[first_brace : last_brace + 1])
            except json.JSONDecodeError:
                pass

        raise OutputParsingError(
            message="Could not extract valid JSON from LLM response",
            detail=f"Raw response: {text[:200]}",
        )

    def validate_output(self, data: dict, output_format: OutputFormat) -> BaseModel:
        """
        Validates extracted JSON against the Pydantic model for the
        requested format.

        OutputFormat.JSON is a special case: its model, CustomFormat, is
        `{"data": dict}` -- but the JSON we actually extracted (matching
        the caller's schema_hint) is the *content* that belongs at that
        `data` key, not something already wrapped in one. So it gets
        constructed as CustomFormat(data=data) rather than
        CustomFormat.model_validate(data); the router unwraps `.data`
        again when building the response so callers see their requested
        shape directly, not double-wrapped.
        """
        model_class = FORMAT_MODEL_MAP[output_format]
        try:
            if output_format == OutputFormat.JSON:
                return model_class(data=data)
            return model_class.model_validate(data)
        except ValidationError as exc:
            raise OutputParsingError(
                message=f"LLM output did not match the expected {output_format.value} schema",
                detail=str(exc),
            ) from exc

    def parse(self, raw_text: str, output_format: OutputFormat) -> BaseModel | None:
        """Main entry point: extract + validate. None for TEXT (nothing to parse)."""
        if output_format == OutputFormat.TEXT:
            return None
        data = self.extract_json(raw_text)
        return self.validate_output(data, output_format)

    def build_retry_prompt(
        self, original_prompt: str, format_instruction: str, error_message: str
    ) -> str:
        """
        A self-contained instructional prompt for a retry attempt -- it
        restates the original request so the model has full context
        again (a fresh /api/chat call has no memory of the first
        attempt), while being much more insistent about format than the
        first try was.
        """
        return (
            f"Your previous response was not valid JSON. Error: {error_message}\n\n"
            f"Please try again. {format_instruction}\n\n"
            "IMPORTANT: Return ONLY the JSON object. Do not include any "
            "text, explanation, or markdown formatting before or after "
            "the JSON.\n\n"
            f"Original request: {original_prompt}"
        )
