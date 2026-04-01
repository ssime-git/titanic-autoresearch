"""Logging and iteration tracking utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def append_log(entry: Dict[str, Any], logs_path: Path) -> None:
    """Append a single JSON line to the iterations log.

    Args:
        entry: Dictionary containing iteration results.
        logs_path: Path to the log file.

    Raises:
        IOError: If file write fails.
    """
    try:
        with open(logs_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Failed to write log entry: {e}")
        raise e


def load_logs(logs_path: Path) -> list:
    """Load all previous iterations from the log file.

    Args:
        logs_path: Path to the log file.

    Returns:
        List of iteration dictionaries, or empty list if no logs exist.

    Raises:
        json.JSONDecodeError: If log file contains invalid JSON.
    """
    try:
        if not logs_path.exists():
            return []
        with open(logs_path) as f:
            logs = []
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
            return logs
    except json.JSONDecodeError as e:
        logger.error(f"Log file contains invalid JSON: {e}")
        raise e
    except IOError as e:
        logger.error(f"Failed to read log file: {e}")
        raise e


def get_best_auc(logs_path: Path) -> float:
    """Get the best AUC from all previous iterations.

    Args:
        logs_path: Path to the log file.

    Returns:
        Best AUC found so far, or 0.5 as default if no logs exist.
    """
    try:
        logs = load_logs(logs_path)
        return max((log["metrics"]["auc_roc"] for log in logs), default=0.5)
    except Exception as e:
        logger.error(f"Failed to get best AUC: {e}")
        return 0.5


def get_current_iteration(logs_path: Path) -> int:
    """Get the next iteration number based on existing logs.

    Args:
        logs_path: Path to the log file.

    Returns:
        Number of completed iterations (0-indexed).
    """
    try:
        logs = load_logs(logs_path)
        return len(logs)
    except Exception as e:
        logger.error(f"Failed to get iteration count: {e}")
        return 0
