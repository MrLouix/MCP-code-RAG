"""Heuristic tagging H1 for source code files."""

from pathlib import Path
from typing import Optional


def tag_h1(file_path: str, lang: Optional[str], chunk: Optional[str] = None) -> list[str]:
    """
    Apply heuristic H1 tagging rules to a file.

    Rules:
    - Always: lang:{language}
    - type:test: path contains test/tests/__tests__ or name ends with _test/_spec/test_
    - type:config: .yaml/.toml/.cfg/.ini/.json/.env/.xml
    - type:doc: .md/.txt or docs/ folder
    - layer:api: api/routes/views/controllers/ or name contains router/route
    - layer:model: models/entities/schemas/db/ or name contains model
    - layer:service: services/lib/core/business/
    - layer:middleware: middleware/ or name contains middleware/interceptor
    - layer:utils: utils/helpers/common/shared/

    Args:
        file_path: Path to the file.
        lang: Detected language from detect_language().
        chunk: Optional chunk content (not used in H1).

    Returns:
        List of tags without duplicates.
    """
    tags = set()
    path = Path(file_path)
    name = path.name
    parts = path.parts  # All path components
    suffix = path.suffix.lower()

    # Always add language tag
    if lang:
        tags.add(f"lang:{lang}")

    # type:test - check path and filename patterns
    path_parts = [p.lower() for p in parts]
    is_test_dir = any(p in ["test", "tests", "__tests__"] for p in path_parts)
    is_test_file = any(name.lower().endswith(pattern) for pattern in ["_test.py", "_test.js", "_test.ts", "_spec.py", "_spec.js", "_test", "_spec"])
    if is_test_dir or is_test_file:
        tags.add("type:test")

    # type:config - config file extensions
    if suffix in [".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".env", ".xml"]:
        tags.add("type:config")
    # Also detect .env files with additional extensions like .env.local, .env.prod
    if ".env" in name:
        tags.add("type:config")

    # type:doc - markdown, text, or docs folder
    if suffix in [".md", ".markdown", ".txt"]:
        tags.add("type:doc")
    if "docs" in parts:
        tags.add("type:doc")

    # layer:api - API-related patterns
    if any(p in path_parts for p in ["api", "routes", "views", "controllers"]):
        tags.add("layer:api")
    name_lower = name.lower()
    if "router" in name_lower or "route" in name_lower:
        tags.add("layer:api")

    # layer:model - model-related patterns
    if any(p in path_parts for p in ["models", "entities", "schemas", "db"]):
        tags.add("layer:model")
    if "model" in name_lower:
        tags.add("layer:model")

    # layer:service - service-related patterns
    if any(p in path_parts for p in ["services", "lib", "core", "business"]):
        tags.add("layer:service")

    # layer:middleware - middleware-related patterns
    if "middleware" in path_parts:
        tags.add("layer:middleware")
    if "middleware" in name_lower or "interceptor" in name_lower:
        tags.add("layer:middleware")

    # layer:utils - utility-related patterns
    if any(p in path_parts for p in ["utils", "helpers", "common", "shared"]):
        tags.add("layer:utils")

    return sorted(list(tags))
