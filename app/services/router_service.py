"""
Smart query routing -- the "intelligence layer" of the gateway.

Two classes live here, split by responsibility:

  1. QueryClassifier: decides WHAT KIND of query this is (simple, complex,
     creative, technical) using a two-tier approach. Tier 1 is a set of
     keyword/length heuristics that run in microseconds and correctly
     classify the vast majority of everyday queries. Tier 2 only kicks in
     when tier 1 itself isn't confident about its answer -- it asks the
     smallest available model to classify the query in a single word.
     This "cheap-first, expensive-only-when-needed" pattern is the same
     idea behind CPU branch prediction or HTTP caching: pay the expensive
     cost only for the hard cases.

  2. ModelRouter: decides WHICH MODEL to actually use, given a query type,
     VRAM state (which models are currently loaded), and the caller's
     preference (manual override vs. auto mode). The key insight driving
     this class is that swapping which model is loaded in VRAM is *slow*
     (several seconds, since Ollama has to evict one model's weights and
     load another's) -- so "the theoretically best model" and "the best
     model to use right now" aren't always the same thing. A loaded model
     that's merely *good* often beats an unloaded model that's *perfect*,
     because the swap cost dwarfs any quality difference for most queries.

Both classes are stateless (no per-request instance data) -- see
RouterService in app/core/dependencies.py, which holds one shared
instance of each as a FastAPI singleton, same pattern as PromptService.
"""

from __future__ import annotations

import structlog

from app.core.exceptions import ModelNotFoundError
from app.models.enums import QueryType

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Tier 1 heuristic classifier -- keyword/phrase sets
# ---------------------------------------------------------------------------
# These sets are intentionally large and a little redundant -- the whole
# point of a keyword heuristic is that it's a blunt instrument that's cheap
# to run on every request. A missed keyword just means a query falls
# through to a later rule (or to the LLM tier), it never causes a wrong
# answer outright.
TECHNICAL_KEYWORDS = {
    "code", "function", "class", "def", "error", "bug", "debug",
    "api", "database", "sql", "python", "javascript", "typescript",
    "java", "rust", "golang", "html", "css", "react", "node",
    "docker", "kubernetes", "git", "compile", "runtime", "syntax",
    "algorithm", "data structure", "binary", "array", "list",
    "hashmap", "stack", "queue", "tree", "graph", "sort", "search",
    "regex", "http", "rest", "graphql", "json", "xml", "yaml",
    "variable", "loop", "recursion", "inheritance", "polymorphism",
    "async", "await", "promise", "callback", "middleware", "endpoint",
    "deploy", "server", "client", "frontend", "backend", "fullstack",
    "framework", "library", "package", "module", "import", "export",
    "test", "unittest", "pytest", "exception", "try", "catch",
}

CREATIVE_KEYWORDS = {
    "write", "story", "poem", "creative", "imagine", "fiction",
    "narrative", "character", "plot", "scene", "dialogue",
    "brainstorm", "ideate", "design", "sketch", "draft",
    "song", "lyrics", "script", "novel", "essay", "blog",
    "metaphor", "describe", "paint", "compose", "invent",
    "fantasy", "adventure", "romance", "mystery", "thriller",
}

# Verbs that, when a query *starts* with them, signal the caller is asking
# for something to be created/generated rather than merely mentioning a
# creative-sounding word in passing (e.g. "write a function" shouldn't
# necessarily out-vote a technical classification just because "write" is
# in CREATIVE_KEYWORDS -- but Rule 3 below only applies this leading-verb
# check anyway, so a bare mention elsewhere in the sentence doesn't count).
CREATIVE_LEADING_VERBS = {"write", "create", "compose", "imagine"}

# Multi-word phrases indicating the caller wants an analytical/comparative
# treatment -- these are checked with substring "in" tests (not word-splits)
# since they're phrases, not single tokens.
COMPLEX_INDICATORS = {
    "compare", "contrast", "analyze", "evaluate", "explain why",
    "pros and cons", "trade-offs", "tradeoffs", "advantages",
    "disadvantages", "implications", "consequences", "impact",
    "relationship between", "difference between", "how does",
    "why does", "what causes", "step by step", "in detail",
    "comprehensive", "thorough", "elaborate", "deep dive",
}

# Short greetings/pleasantries -- always SIMPLE regardless of length or
# any keyword overlap, since these carry no real information content to
# reason about.
GREETINGS = {
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
    "bye", "goodbye", "yo", "sup", "howdy", "cheers",
}

# Below this heuristic confidence, classify_smart() considers the tier-1
# answer too uncertain to trust outright and prefers the (slower, more
# accurate) LLM classifier when one is available.
CONFIDENCE_THRESHOLD = 0.5


class QueryClassifier:
    """
    Two-tier query classifier: fast keyword heuristics first, an LLM
    fallback only for the ambiguous cases the heuristics aren't sure
    about. See module docstring for the cost/accuracy rationale.
    """

    def classify(self, query: str) -> tuple[QueryType, float]:
        """
        Tier 1: pure heuristic classification, no I/O, sub-millisecond.

        Returns (query_type, confidence) where confidence in [0, 1] is
        our own estimate of how sure this heuristic is -- NOT a
        model-calibrated probability, just a simple signal that
        classify_smart() uses to decide whether to escalate to the LLM
        tier.

        Rule 1 (greeting / very short query) is checked first and
        short-circuits immediately, since length is the strongest signal
        available for the "obviously simple" case and nothing else needs
        to be considered once it fires.

        Rules 2-4 (technical / creative / complex) are NOT applied as a
        strict first-match-wins chain, even though each is individually
        worded that way -- a query can legitimately trip more than one
        of them at once (e.g. "Compare SQL and NoSQL databases..." hits
        both Rule 2's technical-keyword count via "sql"/"database" AND
        Rule 4's complex-phrase count via "compare"/"contrast"), and
        whichever rule happens to be checked first would otherwise always
        win regardless of which signal is actually stronger for that
        particular query. So instead every rule that matches computes its
        own confidence, and the highest-confidence match wins -- the
        formulas are unchanged, only the selection between simultaneous
        matches is confidence-based rather than order-based.
        """
        lowered = query.lower().strip()
        words = lowered.split()

        # Rule 1 -- short/simple queries and greetings are the easiest
        # case to get right, so they're checked first and with the
        # highest confidence, short-circuiting before any other rule.
        if lowered in GREETINGS or any(
            lowered.startswith(greeting) for greeting in GREETINGS
        ):
            return QueryType.SIMPLE, 0.99
        # "No keyword matches" here means none of the three category
        # keyword sets, not just technical -- otherwise a short but
        # clearly creative prompt like "Write a poem about the ocean"
        # (29 chars) would get short-circuited to SIMPLE before Rule 3
        # ever gets a chance to see its "write"/"poem" signal, which
        # contradicts what a short creative request actually is.
        has_any_keyword_match = (
            self._count_keyword_matches(lowered, TECHNICAL_KEYWORDS)
            or self._count_keyword_matches(lowered, CREATIVE_KEYWORDS)
            or any(phrase in lowered for phrase in COMPLEX_INDICATORS)
        )
        if len(query) < 30 and not has_any_keyword_match:
            return QueryType.SIMPLE, 0.95

        candidates: list[tuple[QueryType, float]] = []

        # Rule 2 -- technical detection. Two or more distinct technical
        # keywords is a strong signal; exactly one is weaker evidence
        # unless it's specifically code/error/debug-flavored (a query
        # that merely says "python" once might just be making small talk
        # about the language, but "debug" or "error" strongly implies an
        # actual technical question).
        technical_matches = self._count_keyword_matches(lowered, TECHNICAL_KEYWORDS)
        if technical_matches >= 2:
            candidates.append((QueryType.TECHNICAL, min(0.95, 0.6 + technical_matches * 0.1)))
        elif technical_matches == 1 and any(
            k in lowered for k in ("code", "error", "debug")
        ):
            candidates.append((QueryType.TECHNICAL, 0.7))

        # Rule 3 -- creative detection. Requires both a creative keyword
        # AND a leading creative verb, since "write" alone is ambiguous
        # ("write a function" is technical, "write a poem" is creative)
        # but "write" as the very first word combined with any creative
        # keyword elsewhere in the query is a much stronger signal of
        # actual creative intent.
        creative_matches = self._count_keyword_matches(lowered, CREATIVE_KEYWORDS)
        if creative_matches >= 1 and words and words[0] in CREATIVE_LEADING_VERBS:
            candidates.append((QueryType.CREATIVE, min(0.95, 0.7 + creative_matches * 0.1)))

        # Rule 4 -- complex detection. These indicators are multi-word
        # phrases, so membership is checked with substring ("in") tests
        # rather than splitting into individual words.
        complex_matches = sum(1 for phrase in COMPLEX_INDICATORS if phrase in lowered)
        if complex_matches >= 1:
            candidates.append((QueryType.COMPLEX, min(0.9, 0.6 + complex_matches * 0.15)))
        elif len(query) > 100:
            # Long, but no explicit complex phrase fired -- weakly assume
            # it's asking for something involved rather than simple, but
            # with low confidence since we're really just guessing off
            # length alone here.
            candidates.append((QueryType.COMPLEX, 0.6))

        if candidates:
            return max(candidates, key=lambda candidate: candidate[1])

        # Rule 5 -- default: no rule matched at all. This is the
        # lowest-confidence outcome by design, so classify_smart() knows
        # to prefer the LLM tier for these if one is available.
        if len(query) < 80:
            return QueryType.SIMPLE, 0.5
        return QueryType.COMPLEX, 0.4

    @staticmethod
    def _count_keyword_matches(lowered_query: str, keywords: set[str]) -> int:
        """Counts how many distinct keywords from the set appear anywhere
        in the (already-lowercased) query, via substring search -- this
        deliberately also matches keywords that are multi-word phrases
        (e.g. "data structure") which a plain word-split wouldn't catch."""
        return sum(1 for keyword in keywords if keyword in lowered_query)

    async def classify_with_llm(self, query: str, llm_service) -> QueryType:
        """
        Tier 2: ask the smallest, fastest model (gemma3:4b) to classify
        the query itself. Only called when tier 1's confidence is too low
        to trust -- this costs a real network round-trip and inference
        call (roughly 0.5-1s), so it's reserved for genuinely ambiguous
        queries rather than run on every request.

        temperature=0.1 and max_tokens=10 both push toward getting back
        exactly one short, consistent word rather than a rambling
        explanation -- we only need the classification, not the model's
        reasoning about it.
        """
        classification_prompt = f"""Classify this user query into exactly one category.

Categories:
- simple: basic questions, greetings, short factual queries
- complex: multi-part analysis, comparisons, detailed explanations
- creative: writing, storytelling, brainstorming, artistic content
- technical: programming, debugging, code review, technical concepts

Query: "{query}"

Respond with ONLY one word: simple, complex, creative, or technical"""

        result = await llm_service.chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.1,
            max_tokens=10,
        )

        response_text = result.text.strip().lower()
        for query_type in QueryType:
            if query_type.value in response_text:
                return query_type

        # The model responded with something unrecognizable (e.g. it
        # ignored the "one word" instruction) -- COMPLEX is the safest
        # default since its system prompt/temperature are the most
        # generally-applicable of the four (see prompt_service.py).
        logger.warning("llm_classification_unrecognized", raw_response=response_text)
        return QueryType.COMPLEX

    async def classify_smart(
        self, query: str, llm_service
    ) -> tuple[QueryType, float, str]:
        """
        Main entry point combining both tiers.

        Returns (query_type, confidence, method) where method is one of
        "heuristic" (tier 1 was confident enough on its own),
        "llm" (tier 1 was uncertain, tier 2 resolved it), or
        "heuristic-fallback" (tier 1 was uncertain but no llm_service was
        available to escalate to, e.g. it's None in a context that
        doesn't have one) -- this three-way method label is what shows up
        in ResponseMetadata.classification_method so callers can see
        exactly how their query got classified.
        """
        query_type, confidence = self.classify(query)

        if confidence >= CONFIDENCE_THRESHOLD:
            return query_type, confidence, "heuristic"

        if llm_service is not None:
            llm_query_type = await self.classify_with_llm(query, llm_service)
            # The LLM tier doesn't produce its own calibrated confidence
            # score (it just returns a category), so 0.8 is a fixed
            # assumed confidence -- high enough to be trusted over the
            # low-confidence heuristic result, but not as high as the
            # heuristic's own best-case scores, since we have no real
            # signal into how sure the LLM actually was.
            return llm_query_type, 0.8, "llm"

        return query_type, confidence, "heuristic-fallback"


# ---------------------------------------------------------------------------
# Model routing table
# ---------------------------------------------------------------------------
# Maps each QueryType to its ideal ("primary") model plus an ordered list
# of fallbacks to try if the primary isn't loaded/available. Fallback
# order matters: each list is ordered roughly by "how close is this
# model's specialty to the primary's", so a swap-avoiding substitution
# degrades gracefully rather than jumping to something unrelated.
MODEL_ROUTING_TABLE: dict[QueryType, dict[str, list[str] | str]] = {
    QueryType.SIMPLE: {
        "primary": "gemma3:4b",
        "fallbacks": ["phi4-mini:latest", "qwen2.5:7b"],
    },
    QueryType.COMPLEX: {
        "primary": "qwen2.5:7b",
        "fallbacks": ["mistral:7b", "phi4-mini:latest"],
    },
    QueryType.CREATIVE: {
        "primary": "mistral:7b",
        "fallbacks": ["qwen2.5:7b", "gemma3:4b"],
    },
    QueryType.TECHNICAL: {
        "primary": "qwen2.5-coder:7b",
        "fallbacks": ["qwen2.5:7b", "phi4-mini:latest"],
    },
}

# Models small enough to coexist with a 7B model in a typical 8GB VRAM
# budget -- swapping *to* one of these from a 7B (or vice versa) doesn't
# necessarily evict the other, so treating them specially in swap
# decisions reflects real VRAM behavior, not just a routing preference.
SMALL_MODELS = {"gemma3:4b", "phi4-mini:latest"}


class ModelRouter:
    """
    Picks the best model for a query given VRAM state and user
    preference. See module docstring for why "loaded but merely good"
    frequently beats "unloaded but perfect".
    """

    def route(
        self,
        query_type: QueryType,
        preferred_model: str | None,
        loaded_models: list[str],
        available_models: list[str],
        confidence: float = 1.0,
    ) -> str:
        """
        Core routing decision.

        Manual mode (preferred_model set): the caller has made an
        explicit choice, which always wins over any auto-routing logic
        -- we only check that the model actually exists, we never
        second-guess *which* model a manual caller wanted.

        Auto mode (preferred_model is None): walks the routing table for
        this query_type, preferring (in order) the already-loaded primary,
        then -- via should_swap_model -- either an already-loaded fallback
        (to avoid a VRAM swap) or the primary itself if the classification
        is confident enough to justify swapping anyway, then unloaded
        fallbacks in order, finally raising if nothing in the table is
        available at all.

        `confidence` defaults to 1.0 (always worth swapping to the ideal
        model) so existing callers that don't have a meaningful
        confidence score to pass (e.g. manual mode, which never reaches
        this branch anyway) don't need to think about it.
        """
        if preferred_model is not None:
            if preferred_model in available_models:
                return preferred_model
            raise ModelNotFoundError(
                f"Model '{preferred_model}' not found",
                detail=f"Available models: {', '.join(available_models) or 'none'}",
            )

        routing_entry = MODEL_ROUTING_TABLE[query_type]
        primary: str = routing_entry["primary"]  # type: ignore[assignment]
        fallbacks: list[str] = routing_entry["fallbacks"]  # type: ignore[assignment]

        # 2b: primary already resident in VRAM -- the ideal case, no
        # swap needed at all.
        if primary in loaded_models:
            return primary

        if primary in available_models:
            # 2c: primary is installed but not currently loaded. Before
            # accepting the swap cost, check whether any fallback for
            # this query type is *already* loaded. A loaded fallback
            # alone isn't automatically the answer, though -- we defer
            # to should_swap_model to decide whether THIS classification
            # is confident enough that swapping to the ideal model is
            # worth it anyway (e.g. a high-confidence creative query
            # shouldn't silently settle for gemma3:4b just because it
            # happens to be preloaded and also listed as a fallback).
            for fallback in fallbacks:
                if fallback in loaded_models:
                    if self.should_swap_model(fallback, primary, query_type, confidence):
                        return primary
                    return fallback
            # No loaded fallback either -- there's no way to avoid a
            # swap, so take the ideal model and pay the cost.
            return primary

        # 2d: primary isn't even installed in Ollama at all (not just
        # "not loaded") -- walk the fallback chain for the first one
        # that's actually available.
        for fallback in fallbacks:
            if fallback in available_models:
                return fallback

        # 2e: nothing in the whole routing table (primary or any
        # fallback) is available -- there's genuinely no model we can
        # route this query to.
        raise ModelNotFoundError(
            f"No available model found for query type '{query_type.value}'",
            detail=f"Available models: {', '.join(available_models) or 'none'}",
        )

    def get_fallback_chain(
        self, model: str, query_type: QueryType, available_models: list[str]
    ) -> list[str]:
        """
        Ordered list of models to retry with if `model` fails at
        request time (e.g. Ollama returns a connection error mid-call).
        Filtered to only models Ollama actually reports as available,
        so callers never waste a retry attempt on a model that doesn't
        exist.
        """
        fallbacks: list[str] = MODEL_ROUTING_TABLE[query_type]["fallbacks"]  # type: ignore[assignment]
        return [m for m in fallbacks if m in available_models and m != model]

    def should_swap_model(
        self,
        current_model: str,
        target_model: str,
        query_type: QueryType,
        confidence: float,
    ) -> bool:
        """
        Standalone swap-worthiness decision, exposed mainly for logging
        and observability (see chat.py's "query_routed" log event) --
        route() above already makes its own swap-avoidance decision
        inline, but this method captures the same reasoning as an
        explicit, independently-callable policy: is swapping FROM
        current_model TO target_model worth the latency cost, given how
        confident we are in this classification?
        """
        if current_model == target_model:
            # Already there -- nothing to swap.
            return False

        fallbacks: list[str] = MODEL_ROUTING_TABLE[query_type]["fallbacks"]  # type: ignore[assignment]
        if current_model in fallbacks and confidence < 0.8:
            # The model we already have loaded is an acceptable answer
            # for this query type, and we're not confident enough in the
            # classification to justify paying a swap cost for a
            # possibly-wrong "improvement".
            return False

        primary: str = MODEL_ROUTING_TABLE[query_type]["primary"]  # type: ignore[assignment]
        if target_model == primary and confidence >= 0.8:
            # High-confidence classification pointing at the ideal
            # model -- worth the swap.
            return True

        if current_model in SMALL_MODELS:
            # Small models coexist with a 7B in VRAM (see SMALL_MODELS
            # docstring above) -- "swapping" from one doesn't necessarily
            # evict it, so there's no real cost to avoid here.
            return True

        # Default: swap to the optimal model.
        return True
