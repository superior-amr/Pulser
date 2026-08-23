"""Page-aware cleaning: artifacts, headers/footers, front matter, sections.

This module consolidates the notebook's entire Block 8A cleaning stage
(hyphen repair, page-artifact detection, journal headers, repeated-line
header/footer detection, TOC detection, section hierarchy, front-matter
classification, clinical-start detection) into composable, testable units.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import PreprocessingConfig

# ---------------------------------------------------------------------------
# Page preparation (the main entry point)
# ---------------------------------------------------------------------------


@dataclass
class PreparedPage:
    page_number: int
    lines: List[str]

def prepare_pdf_pages_from_docling(
    docling_doc: "DoclingDocument",
    cfg: PreprocessingConfig,
) -> Tuple[List[PreparedPage], List[Tuple[int, str]]]:
    """يمشي على عناصر Docling (headings/paragraphs/tables) بترتيب القراءة
    الصح، ويبني PreparedPage لكل صفحة — بدون أي تخمين regex."""
    prepared: Dict[int, List[str]] = {}
    removed = []
    for item, _level in docling_doc.iterate_items():
        page_no = item.prov[0].page_no if item.prov else None
        if page_no is None:
            continue
        if item.label in ("page_header", "page_footer", "footnote"):
            continue  
        prepared.setdefault(page_no, []).append(item.text)
    pages = [PreparedPage(page_number=p, lines=lines) for p, lines in sorted(prepared.items())]
    return pages, removed
