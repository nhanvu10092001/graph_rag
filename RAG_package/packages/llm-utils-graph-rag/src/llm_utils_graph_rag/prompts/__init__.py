"""Prompt management and template loading utilities for Graph RAG."""

from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).parent


def load_prompt_file(filename: str = "extraction_prompt.md", custom_path: Optional[str] = None) -> str:
    """Load prompt text from a markdown file.

    Args:
        filename: Name of the prompt file inside the prompts directory.
        custom_path: Optional path to an external markdown prompt file.

    Returns:
        The content of the prompt template as a string.
    """
    if custom_path:
        target_path = Path(custom_path)
    else:
        target_path = PROMPTS_DIR / filename

    if not target_path.exists():
        raise FileNotFoundError(f"Prompt file not found at: {target_path.absolute()}")

    return target_path.read_text(encoding="utf-8")