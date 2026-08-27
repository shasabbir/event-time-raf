"""Retired presentation builder.

The previous script encoded July 2026 Event-TimeRAF results and used a slide
generation path that is no longer part of the verified publication workflow.
It is retained as a guard so an obsolete deck cannot be regenerated silently.
"""

from __future__ import annotations


RUN_ID = "20260827T043457543402Z"


raise SystemExit(
    "This legacy builder is disabled because it contains obsolete experiment "
    "content. Rebuild TRACE-RAF_Verified_Presentation.pptx from "
    f"presentation/presenter_notes.md for run {RUN_ID}."
)
