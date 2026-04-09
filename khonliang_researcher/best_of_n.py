"""Self-distillation primitive: generate N candidates, select the best.

Generic best-of-N sampling for any LLM client. Generates `n` candidates
in parallel at higher temperature for diversity, then asks the same model
to select the strongest one.

Used by:
  - researcher.synergize for FR generation diversity
  - developer spec evaluation (planned)
  - any caller that needs cheap quality improvement over single sampling
"""

import asyncio
import json
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


DEFAULT_SELECTION_PROMPT = """\
You are evaluating {n} candidate responses to the same prompt.
Select the candidate that is most accurate, complete, and well-structured.

Output ONLY the candidate number (1-{n}). No explanation, no punctuation.

{candidates}
"""


async def select_best_of_n(
    client: Any,
    prompt: str,
    n: int = 3,
    *,
    system: str = "",
    temperature: float = 0.7,
    selection_temperature: float = 0.1,
    max_tokens: int = 6000,
    selection_max_tokens: int = 10,
    selection_prompt_template: str = DEFAULT_SELECTION_PROMPT,
    selection_system: str = "Select the best candidate. Output only the number.",
    return_candidates: bool = False,
    model: Optional[str] = None,
) -> Union[str, dict]:
    """Generate N candidates in parallel, ask the same model to pick the best.

    Args:
        client: LLM client with async ``generate(prompt, system, temperature,
            max_tokens, model=...)`` method (e.g. khonliang's OllamaClient).
        prompt: Generation prompt for each candidate.
        n: Number of candidates to sample. n=1 short-circuits to a single
            generation at the lower selection_temperature.
        system: System prompt for the generation calls.
        temperature: Temperature for candidate generation. Higher = more
            diverse candidates.
        selection_temperature: Temperature for the selection call. Lower =
            more deterministic ranking.
        max_tokens: Token cap per candidate generation.
        selection_max_tokens: Token cap for the selection response.
        selection_prompt_template: Template with ``{n}`` and ``{candidates}``
            placeholders. Override for domain-specific selection criteria.
        selection_system: System prompt for the selection call.
        return_candidates: When True, return a dict with all candidates and
            the selection metadata. When False (default), return just the
            chosen candidate string.
        model: Optional model override passed to ``client.generate``.

    Returns:
        - When ``return_candidates=False``: the selected candidate text.
        - When ``return_candidates=True``: ``{"selected": int, "candidates": list[str]}``
          where ``selected`` is the 1-indexed winner.
    """
    if n <= 1:
        return await client.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"model": model} if model else {}),
        )

    async def _sample() -> str:
        return await client.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"model": model} if model else {}),
        )

    candidates = list(await asyncio.gather(*[_sample() for _ in range(n)]))
    logger.info("best-of-N: generated %d candidates, selecting best", len(candidates))

    candidate_text = "\n\n".join(
        f"=== CANDIDATE {i + 1} ===\n{c}" for i, c in enumerate(candidates)
    )
    choice_raw = await client.generate(
        prompt=selection_prompt_template.format(n=len(candidates), candidates=candidate_text),
        system=selection_system,
        temperature=selection_temperature,
        max_tokens=selection_max_tokens,
        **({"model": model} if model else {}),
    )

    selected = 0
    try:
        choice = int(str(choice_raw).strip()) - 1
        if 0 <= choice < len(candidates):
            selected = choice
            logger.info("best-of-N: selected candidate %d", selected + 1)
    except (ValueError, TypeError):
        logger.warning(
            "best-of-N: could not parse selection '%s', using candidate 1",
            str(choice_raw).strip(),
        )

    if return_candidates:
        return {
            "selected": selected + 1,
            "candidates": candidates,
        }

    return candidates[selected]


def serialize_candidates(result: dict) -> str:
    """Helper to JSON-encode a return_candidates=True result for transport."""
    return json.dumps(result)
