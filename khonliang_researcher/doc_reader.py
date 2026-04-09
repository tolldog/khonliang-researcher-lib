"""Lightweight, structure-aware reader for local docs.

Reads markdown/text files into structured form without persistence,
distillation, or LLM calls. Used by callers that want spec/milestone
content on demand without going through a full ingestion pipeline.

Capabilities:
  - Read whole file with parsed frontmatter and section index
  - Extract a specific markdown section by heading
  - Parse YAML frontmatter
  - Find FR/spec/milestone IDs (or arbitrary patterns) referenced in a doc
  - Glob for docs under a root directory
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Default reference pattern matches IDs like ``fr_researcher_c6b7dca8``,
# ``fr_khonliang_03f461fa``, etc. Override per call for other ID schemes.
DEFAULT_REFERENCE_PATTERN = r"fr_[a-zA-Z0-9_-]+"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class DocContent:
    """Parsed content of a single document."""

    path: str
    text: str
    frontmatter: Dict[str, object] = field(default_factory=dict)
    sections: Dict[str, str] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)


class LocalDocReader:
    """Read local files with structure awareness. No persistence, no LLM."""

    def __init__(self, reference_pattern: str = DEFAULT_REFERENCE_PATTERN):
        self.reference_pattern = reference_pattern

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, path: str) -> DocContent:
        """Read a file and return its structured content.

        Frontmatter is parsed if present, sections are indexed by heading
        text, and references matching ``reference_pattern`` are extracted.
        """
        text = Path(path).read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(text)
        sections = self._index_sections(body)
        references = self._find_references(body, self.reference_pattern)

        return DocContent(
            path=str(path),
            text=text,
            frontmatter=frontmatter,
            sections=sections,
            references=references,
        )

    def extract_section(self, path: str, heading: str) -> str:
        """Return just one markdown section by heading text.

        Match is case-insensitive and ignores leading ``#`` markers in the
        argument. Returns an empty string if the heading is not found.
        """
        doc = self.read(path)
        return self._lookup_section(doc.sections, heading)

    def parse_frontmatter(self, path: str) -> Dict[str, object]:
        """Parse and return YAML frontmatter from a markdown file."""
        text = Path(path).read_text(encoding="utf-8")
        frontmatter, _ = self._split_frontmatter(text)
        return frontmatter

    def find_references(
        self, path: str, pattern: Optional[str] = None
    ) -> List[str]:
        """Find IDs referenced in a doc. Defaults to FR ID pattern."""
        text = Path(path).read_text(encoding="utf-8")
        return self._find_references(text, pattern or self.reference_pattern)

    def glob_docs(self, root: str, pattern: str = "**/*.md") -> List[str]:
        """Find all docs matching a glob pattern under root."""
        return [str(p) for p in sorted(Path(root).glob(pattern)) if p.is_file()]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[Dict[str, object], str]:
        match = _FRONTMATTER_RE.match(text)
        if not match:
            return {}, text

        try:
            data = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as e:
            logger.warning("Failed to parse frontmatter: %s", e)
            data = {}

        if not isinstance(data, dict):
            data = {}

        body = text[match.end():]
        return data, body

    @staticmethod
    def _index_sections(body: str) -> Dict[str, str]:
        """Index markdown sections by heading text.

        Sections include their heading line and span until the next heading
        of the same or higher level (or end of file). When the same heading
        text appears multiple times, later occurrences are suffixed with
        ``#2``, ``#3``, etc.
        """
        sections: Dict[str, str] = {}
        matches = list(_HEADING_RE.finditer(body))
        if not matches:
            return sections

        for i, match in enumerate(matches):
            heading_text = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            content = body[start:end].rstrip()

            key = heading_text
            count = 2
            while key in sections:
                key = f"{heading_text}#{count}"
                count += 1
            sections[key] = content

        return sections

    @staticmethod
    def _lookup_section(sections: Dict[str, str], heading: str) -> str:
        """Find a section by heading, case-insensitively, ignoring leading #."""
        target = heading.lstrip("#").strip().lower()
        for key, content in sections.items():
            if key.split("#", 1)[0].strip().lower() == target:
                return content
        return ""

    @staticmethod
    def _find_references(text: str, pattern: str) -> List[str]:
        """Return unique references in document order."""
        seen: Dict[str, None] = {}
        for match in re.finditer(pattern, text):
            seen.setdefault(match.group(0), None)
        return list(seen.keys())
