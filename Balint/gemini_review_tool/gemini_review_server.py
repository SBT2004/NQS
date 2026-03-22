from pathlib import Path
import os

from codex_gemini_review.server import main


if __name__ == "__main__":
    scope_root = Path(__file__).resolve().parents[1]
    os.chdir(scope_root)
    os.environ.setdefault("GEMINI_REVIEW_SCOPE_ROOT", str(scope_root))
    os.environ.setdefault("GEMINI_REVIEW_REPO_ROOT", str(scope_root.parent))
    main()
