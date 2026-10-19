import shutil

import pytest

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
