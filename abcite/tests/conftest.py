"""Fast tests never auto-download CLIP or a VLM. Live CLIP/VLM tests set the flags themselves."""

from __future__ import annotations

import os

os.environ.setdefault("ABCITE_CLIP", "0")
os.environ.setdefault("ABCITE_VLM", "0")
os.environ.setdefault("ABCITE_PMC_GALLERY", "0")
