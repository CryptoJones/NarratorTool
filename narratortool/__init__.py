"""NarratorTool — narrate documents to MP3 with local neural TTS."""
__version__ = "0.1.0"

from .document import Chapter, Document  # noqa: E402
from .pipeline import NarrationResult, narrate_file  # noqa: E402

__all__ = ["Chapter", "Document", "NarrationResult", "narrate_file", "__version__"]
