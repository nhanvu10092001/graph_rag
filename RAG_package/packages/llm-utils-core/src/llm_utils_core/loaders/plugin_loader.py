import importlib.metadata
import sys
from typing import Dict, Type

from ..abstracts import TaskPlugin
from ..logger import logger


def load_plugins() -> Dict[str, Type[TaskPlugin]]:
    """Load all registered plugins using entry points."""
    logger.info("Starting to load plugins")
    plugins = {}

    try:
        # Handle both old and new entry_points API
        if sys.version_info >= (3, 10):
            # Use new API for Python 3.10+
            logger.info("Using Python 3.10+ entry_points API")
            entry_points_result = importlib.metadata.entry_points()
            if hasattr(entry_points_result, "select"):
                entry_points = entry_points_result.select(group="llm_utils.plugins")
            else:
                entry_points = entry_points_result.get("llm_utils.plugins", [])
        else:
            # Fallback for Python 3.9
            logger.info("Using Python 3.9 entry_points API")
            try:
                entry_points = importlib.metadata.entry_points()["llm_utils.plugins"]
            except KeyError:
                entry_points = []

        for entry_point in entry_points:
            logger.info(f"Loading plugin: {entry_point.name}")
            plugin_class = entry_point.load()
            plugins[entry_point.name] = plugin_class
            logger.info(f"Successfully loaded plugin: {entry_point.name}")
    except Exception as e:
        logger.error(f"Failed to load plugins: {e}")

    logger.info(f"Loaded {len(plugins)} plugins")
    return plugins
