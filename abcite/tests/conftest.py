"""Fast tests never auto-download CLIP. Live CLIP tests set ABCITE_CLIP themselves."""

from __future__ import annotations

import os

os.environ.setdefault("ABCITE_CLIP", "0")
