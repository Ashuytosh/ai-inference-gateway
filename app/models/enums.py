"""
Shared domain enums.

QueryType lives here rather than in requests.py or responses.py because
it isn't specific to either request validation or response shaping --
it's a classification concept used by the prompt engineering pipeline
(app/services/prompt_service.py, this phase) and, starting Phase 5, the
smart query router (app/services/router_service.py) too.
"""

from enum import Enum


class QueryType(str, Enum):
    """
    The four categories every incoming prompt gets classified into.

    Inheriting from `str` (not just Enum) means QueryType.SIMPLE ==
    "simple" is True, and it JSON-serializes as the plain string "simple"
    instead of the ugly "QueryType.SIMPLE" -- useful since this ends up
    directly in API responses (ResponseMetadata.query_type) and as dict
    keys we look up with plain strings in a few places.
    """

    SIMPLE = "simple"
    COMPLEX = "complex"
    CREATIVE = "creative"
    TECHNICAL = "technical"
