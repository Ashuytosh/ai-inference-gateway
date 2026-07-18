"""
Prompt engineering service.

Prompt engineering is about HOW you talk to an LLM, not just WHAT you
ask it -- the same underlying question, framed differently, produces
noticeably different quality answers. This service automates that
framing so callers (app/routers/chat.py) don't need to know any prompt
engineering themselves; they just classify a query as one of four
QueryType values and this service does the rest:

  1. Picks a system prompt that sets the model's overall persona/behavior
     for that kind of query (get_system_prompt).
  2. Rewrites the user's raw prompt with a strategy suited to that query
     type -- e.g. asking a complex question to be answered step by step
     (apply_strategy).
  3. Optionally injects few-shot examples (a worked example of a good
     answer) before the real question, which tends to anchor the model's
     output format and quality (get_few_shot_examples).
  4. Recommends a temperature suited to the query type, since "creative
     writing" and "precise technical answer" want very different amounts
     of randomness (get_recommended_temperature).

build_messages() is the single entry point that ties 1-3 together into
the exact `messages` list Ollama's /api/chat expects.
"""

import structlog

from app.models.enums import QueryType

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# System prompt templates -- one persona/instruction-set per query type.
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS: dict[QueryType, str] = {
    QueryType.SIMPLE: (
        "You are a helpful assistant. Give clear, concise answers. "
        "Keep responses brief and to the point. Avoid unnecessary detail "
        "unless the user asks for more."
    ),
    QueryType.COMPLEX: (
        "You are a knowledgeable assistant skilled at breaking down complex "
        "topics. Think through problems step by step. Consider multiple "
        "perspectives. Provide thorough, well-structured explanations. "
        "Use examples to illustrate key points."
    ),
    QueryType.CREATIVE: (
        "You are a creative assistant with a vivid imagination. Be "
        "expressive, original, and engaging. Use rich language and "
        "interesting perspectives. Don't be afraid to think outside "
        "the box and offer unique angles."
    ),
    QueryType.TECHNICAL: (
        "You are a senior software engineer and technical expert. Be "
        "precise and accurate. Use proper technical terminology. Include "
        "code examples when relevant. Explain your reasoning. Point out "
        "edge cases, potential issues, and best practices."
    ),
}


# ---------------------------------------------------------------------------
# Strategy display names -- how each QueryType's enhancement shows up in
# ResponseMetadata.prompt_strategy, so the UI/API caller can see exactly
# which prompting technique was used for their request.
# ---------------------------------------------------------------------------
STRATEGY_NAMES: dict[QueryType, str] = {
    QueryType.SIMPLE: "direct",
    QueryType.COMPLEX: "chain-of-thought",
    QueryType.CREATIVE: "creative-enhancement",
    QueryType.TECHNICAL: "technical-precision",
}


# ---------------------------------------------------------------------------
# Recommended temperature per query type -- used only when the caller left
# temperature at the application default (see chat.py's _resolve_temperature),
# since an explicit user-chosen temperature always wins.
# ---------------------------------------------------------------------------
RECOMMENDED_TEMPERATURES: dict[QueryType, float] = {
    QueryType.SIMPLE: 0.3,  # low randomness -- consistent, factual answers
    QueryType.COMPLEX: 0.5,  # balanced -- some room for reasoning variation
    QueryType.CREATIVE: 0.9,  # high randomness -- encourages creativity
    QueryType.TECHNICAL: 0.2,  # very low -- precision matters most
}


# ---------------------------------------------------------------------------
# Few-shot examples -- a worked (question, ideal-answer) pair shown to the
# model before the real question. This "anchors" the model on the format
# and depth we want, which plain instructions alone often don't achieve as
# reliably. Only technical queries have an example defined for now; any
# QueryType without one just gets an empty list (see get_few_shot_examples).
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: dict[QueryType, list[dict[str, str]]] = {
    QueryType.TECHNICAL: [
        {
            "role": "user",
            "content": "What is a Python decorator?",
        },
        {
            "role": "assistant",
            "content": (
                "A decorator is a function that wraps another function to "
                "extend its behavior without modifying it.\n\n"
                "```python\n"
                "def timer(func):\n"
                "    def wrapper(*args, **kwargs):\n"
                "        start = time.time()\n"
                "        result = func(*args, **kwargs)\n"
                "        print(f'{func.__name__} took {time.time()-start:.2f}s')\n"
                "        return result\n"
                "    return wrapper\n\n"
                "@timer\n"
                "def slow_function():\n"
                "    time.sleep(1)\n"
                "```\n\n"
                "Key points:\n"
                "- The `@decorator` syntax is shorthand for `func = decorator(func)`\n"
                "- `functools.wraps` preserves the original function's metadata\n"
                "- Common uses: logging, timing, authentication, caching"
            ),
        },
    ],
}


class PromptService:
    """
    Stateless service (no instance fields) -- all the "state" here is the
    module-level template dicts above, which are the same for every
    request. A single shared instance is still used (see
    app/core/dependencies.py) purely for consistency with the other
    services and to leave room for future state (e.g. template caching)
    without changing callers.
    """

    def get_system_prompt(
        self, query_type: QueryType, custom_system_prompt: str | None
    ) -> str:
        """A user-supplied system prompt always overrides the template --
        they're explicitly telling the model how to behave, which is a
        stronger signal than our generic per-category persona."""
        if custom_system_prompt:
            return custom_system_prompt
        return SYSTEM_PROMPTS[query_type]

    def apply_strategy(self, prompt: str, query_type: QueryType) -> str:
        """
        Rewrites the user's raw prompt to steer the model toward the kind
        of answer that query type needs. Each non-SIMPLE branch appends
        an instruction suffix rather than replacing the prompt, so the
        user's actual question is always preserved verbatim at the start.
        """
        if query_type == QueryType.SIMPLE:
            # Simple queries don't need fancy prompting -- adding
            # instructions would actually make the response unnecessarily
            # verbose for what should be a quick answer.
            return prompt

        if query_type == QueryType.COMPLEX:
            # Chain-of-thought (CoT) prompting: research (Wei et al.,
            # 2022, "Chain-of-Thought Prompting Elicits Reasoning in
            # Large Language Models") shows explicitly asking a model to
            # reason step by step significantly improves answer quality
            # on multi-part/reasoning-heavy questions.
            return (
                f"{prompt}\n\nPlease think through this step by step:\n"
                "1. First, identify the key aspects of this question\n"
                "2. Then, analyze each aspect\n"
                "3. Finally, provide a comprehensive answer"
            )

        if query_type == QueryType.CREATIVE:
            # Creative tasks benefit from explicit permission to be
            # creative -- without it, models tend toward safe, generic
            # responses since "correct and unsurprising" is their default
            # bias absent other instruction.
            return (
                f"{prompt}\n\nBe creative and original in your response. "
                "Feel free to use metaphors, analogies, or unexpected "
                "perspectives to make your answer engaging and memorable."
            )

        # TECHNICAL: structured, actionable coverage -- this checklist-style
        # suffix ensures the model doesn't skip the parts a developer
        # actually needs (not just "what" but "why" and "what could go
        # wrong").
        return (
            f"{prompt}\n\nProvide a precise technical answer. Include:\n"
            "- Code examples if applicable\n"
            "- Explanation of how and why it works\n"
            "- Common pitfalls or edge cases to watch out for\n"
            "- Best practices"
        )

    def get_few_shot_examples(self, query_type: QueryType) -> list[dict[str, str]]:
        return FEW_SHOT_EXAMPLES.get(query_type, [])

    def build_messages(
        self,
        prompt: str,
        query_type: QueryType,
        custom_system_prompt: str | None,
    ) -> list[dict[str, str]]:
        """
        Main entry point: assembles the full Ollama-format messages list
        -- [system] + [few-shot examples, if any] + [user].

        When the caller supplied their own system_prompt, they've taken
        explicit control of the model's behavior -- in that case we skip
        apply_strategy() entirely (the user's raw prompt goes through
        unmodified) rather than layering our own instructions on top of
        theirs, which could easily conflict with what they asked for
        (e.g. asking for "pirate speak" answers, then silently appending
        a chain-of-thought instruction block would fight that framing).
        """
        system_prompt = self.get_system_prompt(query_type, custom_system_prompt)

        if custom_system_prompt:
            enhanced_prompt = prompt
        else:
            enhanced_prompt = self.apply_strategy(prompt, query_type)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self.get_few_shot_examples(query_type))
        messages.append({"role": "user", "content": enhanced_prompt})

        logger.info(
            "prompt_engineered",
            query_type=query_type.value,
            original_length=len(prompt),
            final_length=len(enhanced_prompt),
            system_prompt_source="custom" if custom_system_prompt else "template",
        )

        return messages

    def get_recommended_temperature(self, query_type: QueryType) -> float:
        return RECOMMENDED_TEMPERATURES[query_type]
