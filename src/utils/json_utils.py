"""Utilities for JSON serialization."""

import numpy as np
from typing import Any, Dict, List, Union


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def save_json_safe(data: Union[Dict, List], file_path: str, **kwargs):
    """Save data to JSON file with numpy type conversion."""
    import json
    
    safe_data = convert_to_json_serializable(data)
    
    with open(file_path, 'w') as f:
        json.dump(safe_data, f, **kwargs)