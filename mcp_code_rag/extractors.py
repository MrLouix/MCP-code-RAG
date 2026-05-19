"""Symbol extraction and language detection for source code files."""

import ast
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SymbolLocation:
    """Location information for a symbol in source code."""

    file_path: str
    start_line: int
    end_line: int


@dataclass
class FunctionDescriptor:
    """Complete description of a function or method."""

    language: str
    kind: str  # "function" | "method" | "constructor"
    name: str
    namespace: Optional[str] = None  # Module/package
    signature: str = ""
    description: Optional[str] = None  # Docstring
    location: Optional[SymbolLocation] = None
    parameters: list[dict] = field(default_factory=list)
    returns: Optional[list[dict]] = None
    visibility: Optional[str] = None  # "public" | "private" | "protected"
    is_static: bool = False
    is_async: bool = False
    code: Optional[str] = None  # Full source code
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ClassDescriptor:
    """Complete description of a class, interface, or struct."""

    language: str
    kind: str  # "class" | "interface" | "struct" | "type"
    name: str
    namespace: Optional[str] = None
    full_name: str = ""
    signature: str = ""
    description: Optional[str] = None
    location: Optional[SymbolLocation] = None
    fields: list[dict] = field(default_factory=list)
    methods: list[FunctionDescriptor] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def detect_language(file_path: str) -> Optional[str]:
    """
    Detect programming language from file extension.

    Maps file extensions to normalized language identifiers.
    Returns None for special files like .lock, .min.js, and .DS_Store.

    Args:
        file_path: Path to the source file.

    Returns:
        Normalized language string or None if unrecognized/excluded.
    """
    path = Path(file_path)
    name = path.name
    suffix = path.suffix.lower()

    # Exclude special files (lock files: Gemfile.lock, Cargo.lock, package-lock.json)
    if name == ".DS_Store" or "-lock" in name or name.endswith(".lock") or name.endswith(".min.js"):
        return None

    # Language mapping
    language_map = {
        ".py": "python",
        ".go": "go",
        ".js": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".cs": "csharp",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".c": "cpp",
        ".h": "cpp",
        ".cpp": "cpp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".md": "doc",
        ".markdown": "doc",
        ".txt": "doc",
        ".yaml": "config",
        ".yml": "config",
        ".toml": "config",
        ".json": "config",
        ".cfg": "config",
        ".ini": "config",
        ".env": "config",
        ".xml": "config",
    }

    return language_map.get(suffix)


def is_excluded(path: str, patterns: list[str]) -> bool:
    """
    Check if a path matches any exclusion patterns.

    Uses fnmatch for pattern matching (supports *, ?, [seq], [!seq]).

    Args:
        path: File or directory path to check.
        patterns: List of fnmatch patterns to match against.

    Returns:
        True if path matches any pattern, False otherwise.
    """
    path_obj = Path(path)

    for pattern in patterns:
        # Match against the full path
        if fnmatch.fnmatch(str(path_obj), pattern):
            return True
        # Also match against individual path components
        for part in path_obj.parts:
            if fnmatch.fnmatch(part, pattern):
                return True

        # Special handling for patterns like *.lock to match *-lock.json
        # If pattern is *.ext, also try matching *-ext.* files
        if pattern.startswith("*."):
            ext = pattern[2:]  # e.g., "lock" from "*.lock"
            name = path_obj.name
            if f"-{ext}." in name or f"-{ext}" in name or name.endswith(f".{ext}"):
                return True

    return False


def extract_imports_python(source: str) -> list[str]:
    """
    Extract import statements from Python source code using AST.

    Handles both 'import' and 'from ... import' statements.

    Args:
        source: Python source code as string.

    Returns:
        List of import strings (module names).
    """
    imports = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def extract_imports_generic(source: str, lang: str) -> list[str]:
    """
    Extract import statements using regex patterns for various languages.

    Supports: JavaScript/TypeScript (import/from), Go (import ""),
    C# (using), and basic patterns for other languages.

    Args:
        source: Source code as string.
        lang: Language identifier ("javascript", "typescript", "go", "csharp", etc).

    Returns:
        List of import strings.
    """
    imports = []

    if lang in ("javascript", "typescript"):
        # Match: import ... from "module"
        # Match: import "module"
        pattern = r'(?:import\s+.*?\s+from\s+["\']([^"\']+)["\']|import\s+["\']([^"\']+)["\'])'
        matches = re.findall(pattern, source)
        for match in matches:
            # Each match is a tuple; take the non-empty group
            imp = match[0] or match[1]
            if imp:
                imports.append(imp)

    elif lang == "go":
        # Match: import "module"
        single_pattern = r'import\s+"([^"]+)"'
        matches = re.findall(single_pattern, source)
        imports.extend(matches)

        # Match: import ( "module" ... )
        grouped_pattern = r'import\s+\(\s*([^)]*)\)'
        grouped_matches = re.findall(grouped_pattern, source, re.DOTALL)
        for group in grouped_matches:
            lines = group.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    imports.append(line.strip('"'))

    elif lang == "csharp":
        # Match: using NameSpace;
        pattern = r'using\s+([^\s;]+)'
        matches = re.findall(pattern, source)
        imports.extend(matches)

    else:
        # Generic fallback for unsupported languages
        # Look for common import patterns
        patterns = [
            r'import\s+["\']([^"\']+)["\']',  # import "module"
            r'from\s+["\']([^"\']+)["\']',  # from "module"
            r'using\s+([^\s;]+)',  # using Module
        ]
        for pattern in patterns:
            matches = re.findall(pattern, source)
            imports.extend(matches)

    return list(set(imports))  # Return unique imports


def resolve_import_to_path(
    import_str: str,
    source_file: str,
    workspace_root: str,
    lang: str,
) -> Optional[str]:
    """
    Resolve an import statement to an actual file path in the workspace.

    Args:
        import_str: Import string (e.g., "os", "pathlib", "@types/node").
        source_file: Path to the file containing the import.
        workspace_root: Root directory of the workspace.
        lang: Language identifier.

    Returns:
        Resolved file path if found, None otherwise.
    """
    source_path = Path(source_file).resolve()
    workspace_path = Path(workspace_root).resolve()

    if lang == "python":
        return _resolve_python_import(import_str, source_path, workspace_path)
    elif lang in ("javascript", "typescript"):
        return _resolve_js_import(import_str, source_path, workspace_path)
    elif lang == "go":
        return _resolve_go_import(import_str, source_path, workspace_path)
    elif lang == "csharp":
        return _resolve_csharp_import(import_str, workspace_path)
    else:
        return None


def _resolve_python_import(
    import_str: str,
    source_path: Path,
    workspace_path: Path,
) -> Optional[str]:
    """Resolve Python import to file path."""
    # Handle relative imports
    if import_str.startswith("."):
        # Count leading dots
        dots = len(import_str) - len(import_str.lstrip("."))
        rest = import_str[dots:]

        # Navigate up the directory tree
        current = source_path.parent
        for _ in range(dots - 1):
            current = current.parent

        if rest:
            parts = rest.split(".")
            candidate = current / "/".join(parts[:-1]) / f"{parts[-1]}.py"
            if candidate.exists():
                return str(candidate)
            candidate = current / "/".join(parts) / "__init__.py"
            if candidate.exists():
                return str(candidate)
        return None

    # Handle absolute imports
    parts = import_str.split(".")
    candidates = [
        workspace_path / "/".join(parts) / "__init__.py",
        workspace_path / "/".join(parts) / f"{parts[-1]}.py",
        workspace_path / "/".join(parts[:-1]) / f"{parts[-1]}.py",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    return None


def _resolve_js_import(
    import_str: str,
    source_path: Path,
    workspace_path: Path,
) -> Optional[str]:
    """Resolve JavaScript/TypeScript import to file path."""
    # Handle relative imports
    if import_str.startswith("."):
        base_dir = source_path.parent
        resolved = (base_dir / import_str).resolve()

        # Try extensions
        for ext in [".js", ".ts", ".tsx", ".jsx", ""]:
            candidate = Path(str(resolved) + ext)
            if candidate.exists() and candidate.is_file():
                return str(candidate)

            # Try index file
            if resolved.is_dir():
                index_candidate = resolved / f"index{ext}"
                if index_candidate.exists():
                    return str(index_candidate)

        return None

    # Handle node_modules and absolute imports
    # Try node_modules first
    node_modules = workspace_path / "node_modules" / import_str
    for ext in [".js", ".ts", ".tsx", "", "/index.js", "/index.ts"]:
        candidate = Path(str(node_modules) + ext)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    return None


def _resolve_go_import(
    import_str: str,
    source_path: Path,
    workspace_path: Path,
) -> Optional[str]:
    """Resolve Go import to file path."""
    # Go imports are based on GOPATH/GOROOT
    # For local resolution, we check within workspace
    parts = import_str.split("/")
    candidate = workspace_path / "/".join(parts) / "main.go"

    if candidate.exists():
        return str(candidate)

    # Try to find any .go file in the directory
    candidate_dir = workspace_path / "/".join(parts)
    if candidate_dir.exists() and candidate_dir.is_dir():
        go_files = list(candidate_dir.glob("*.go"))
        if go_files:
            return str(go_files[0])

    return None


def _resolve_csharp_import(
    import_str: str,
    workspace_path: Path,
) -> Optional[str]:
    """Resolve C# using statement to file path (searches for namespace definition)."""
    # C# namespaces are typically defined in .cs files
    # Search for files containing the namespace
    for cs_file in workspace_path.glob("**/*.cs"):
        try:
            with open(cs_file, "r", encoding="utf-8") as f:
                content = f.read()
                if f"namespace {import_str}" in content:
                    return str(cs_file)
        except (OSError, UnicodeDecodeError):
            continue

    return None
