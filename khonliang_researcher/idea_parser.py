"""LLM role for decomposing informal text into researchable components.

Takes a LinkedIn post, tweet, blog snippet, freeform thought, spec draft,
or PR description and produces structured claims + search queries that
downstream pipelines can act on.

Subclass to override DEFAULT_PROMPT, model selection, or result shape.
"""

import logging
import re
from typing import Any, Dict, Optional

from khonliang.pool import ModelPool
from khonliang.roles.base import BaseRole

logger = logging.getLogger(__name__)


DEFAULT_IDEA_PROMPT = """\
You decompose informal text into researchable components. The input may be:
- A LinkedIn or Twitter post
- A blog excerpt or newsletter snippet
- A freeform thought or hypothesis
- A conference talk summary
- A spec draft, design note, or PR description

Your job is to identify the core claims and generate search queries that
would find relevant academic literature.

Output ONLY valid JSON with this schema:

```json
{
  "title": "Short label for this idea (5-10 words)",
  "source_type": "linkedin|twitter|blog|freeform|spec|other",
  "claims": [
    "A specific, testable claim made or implied in the text"
  ],
  "search_queries": [
    "A query suitable for arxiv or Semantic Scholar search"
  ],
  "keywords": ["keyword1", "keyword2"]
}
```

Rules:
- title: a concise label, not the full text
- source_type: best guess from the tone and format
- claims: extract 1-5 distinct claims. Each should be a single sentence
  stating something that could be supported or contradicted by literature.
  Rephrase vague statements into testable claims.
- search_queries: generate 2-6 queries. Each should be 3-8 words, academic
  in tone, suitable for paper search. Cover different angles of the claims.
  Prefer specific technical terms over generic ones.
- keywords: 3-8 specific terms for indexing (methods, models, techniques
  mentioned or implied)
"""


def clean_for_json(text: str) -> str:
    """Strip content that confuses LLM JSON generation.

    Removes: math notation, LaTeX, unicode symbols, excessive whitespace.
    Keeps: readable English text, numbers, basic punctuation. The result
    is collapsed to a single line — paragraph breaks are not preserved.
    """
    text = re.sub(r"\$\$.*?\$\$", "[math]", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "[math]", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class BaseIdeaParser(BaseRole):
    """Decompose informal text into claims and search queries.

    Override DEFAULT_PROMPT or pass system_prompt to customize the
    decomposition style. Override _select_model_for_text() to change
    the model-by-length heuristic.
    """

    DEFAULT_PROMPT: str = DEFAULT_IDEA_PROMPT

    # Length thresholds for model selection. Override in subclasses to
    # match your model pool.
    SHORT_MODEL: str = "llama3.2:3b"
    LONG_MODEL: str = "qwen2.5:7b"
    SHORT_THRESHOLD: int = 2_000
    MAX_INPUT_CHARS: int = 15_000

    def __init__(
        self,
        model_pool: ModelPool,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            role="idea_parser",
            model_pool=model_pool,
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            **kwargs,
        )

    def _select_model_for_text(self, text: str) -> str:
        """Pick a model based on input length. Override for custom routing."""
        return self.LONG_MODEL if len(text) > self.SHORT_THRESHOLD else self.SHORT_MODEL

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """Normalize the LLM response into the expected dict shape."""
        if not isinstance(result, dict):
            logger.warning("Unexpected idea parser output: %s", type(result))
            return {"success": False, "error": "Unexpected output format"}

        return {
            "title": result.get("title", "Untitled idea"),
            "source_type": result.get("source_type", "freeform"),
            "claims": result.get("claims", []),
            "search_queries": result.get("search_queries", []),
            "keywords": result.get("keywords", []),
            "success": True,
        }

    async def handle(
        self,
        message: str,
        session_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decompose `message` into claims and search queries."""
        text = clean_for_json(message[: self.MAX_INPUT_CHARS])
        model = self._select_model_for_text(message)

        prompt = f"Decompose this into researchable claims and search queries:\n\n{text}"

        try:
            result = await self.client.generate_json(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.3,
                max_tokens=2000,
                model=model,
            )
        except Exception as e:
            logger.error("Idea parsing failed with %s: %s", model, e)
            return {"success": False, "error": str(e)}

        return self._normalize_result(result)
