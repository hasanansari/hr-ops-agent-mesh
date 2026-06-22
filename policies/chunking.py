"""Chunking strategy for the HR policy document.

Splits on level-2 markdown headings (`## ...`) rather than fixed-size
sliding windows. The policy doc is already organized into short,
self-contained sections (leave, overtime, payroll tiers, probation,
training, performance) -- each section is one coherent unit of meaning, at
roughly 100-180 words. A fixed-size window (e.g. 200 characters with
overlap) would risk cutting a section in half mid-sentence, separating a
number (e.g. "$5,000") from the rule it belongs to. Heading-based chunking
guarantees every chunk is a complete, self-contained policy statement --
exactly what the retriever needs to hand to the model as a citable,
ungarbled excerpt.
"""

from __future__ import annotations

import re
from pathlib import Path

POLICY_PATH = Path(__file__).parent / "hr_policy.md"

_HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def load_chunks(path: Path = POLICY_PATH) -> list[dict]:
    text = path.read_text()
    matches = list(_HEADING_RE.finditer(text))

    chunks = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append({"chunk_id": i, "title": title, "text": body})
    return chunks
