"""Code chunking for AST-based and regex-based extraction."""

import ast
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from mcp_code_rag.config import AppConfig
from mcp_code_rag.extractors import detect_language


@dataclass
class CodeChunk:
    """Atomic unit of code indexed in the vector database."""

    # Symbol metadata
    type: str  # "function" | "method" | "class" | "interface" | "module" | "file" | "config" | "doc"
    name: str  # Symbol name or filename
    package: str  # Module/package name
    language: str  # "python" | "go" | "typescript" | ...

    # Source location
    file_path: str  # Absolute path
    start_line: int  # Line number (1-indexed)
    end_line: int  # Line number (1-indexed)
    selection_start: int  # Symbol name start line
    selection_end: int  # Symbol name end line

    # Content
    signature: str  # Function/class declaration
    docstring: str = ""  # Documentation (max 2000 chars)
    code: str = ""  # Full source code

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_embed_text(self, max_length: int = 8000) -> str:
        """
        Build embedding text: signature + docstring + code, truncated to max_length.

        Returns:
            Embedding text for vector database.
        """
        parts = [self.signature]
        if self.docstring:
            parts.append(self.docstring)
        if self.code:
            parts.append(self.code)

        text = "\n".join(parts)
        return text[:max_length] if len(text) > max_length else text

    def to_chromadb_metadata(self) -> dict:
        """
        Serialize to JSON-safe metadata dict for ChromaDB.

        Returns:
            Dictionary with string/number values (no nested objects).
        """
        result = {
            "type": self.type,
            "name": self.name,
            "package": self.package,
            "language": self.language,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "selection_start": self.selection_start,
            "selection_end": self.selection_end,
            "signature": self.signature,
            "docstring": self.docstring,
        }

        # Flatten metadata dict into ChromaDB-safe fields
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                result[f"meta_{key}"] = value
            elif isinstance(value, (list, dict)):
                result[f"meta_{key}"] = json.dumps(value)

        return result


def chunk_python_file(source: str, file_path: str, package: str) -> list[CodeChunk]:
    """
    Extract chunks from Python source using AST parsing.

    Extracts: functions, async functions, classes (with their methods), and module-level code.

    Args:
        source: Python source code as string.
        file_path: Absolute path to the file.
        package: Package/module name.

    Returns:
        List of CodeChunk objects.
    """
    chunks = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    lines = source.split("\n")

    # Extract module-level docstring if present
    module_docstring = ast.get_docstring(tree) or ""

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(
                _extract_function_chunk(
                    node, lines, file_path, package, module_docstring
                )
            )
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_extract_class_chunks(node, lines, file_path, package))

    return chunks


def _extract_function_chunk(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    file_path: str,
    package: str,
    module_docstring: str = "",
) -> list[CodeChunk]:
    """Extract a function or async function as a chunk."""
    is_async = isinstance(node, ast.AsyncFunctionDef)
    signature = _build_signature(node, is_async)
    docstring = ast.get_docstring(node) or ""

    # Get source code lines for this function
    start_line = node.lineno  # 1-indexed
    end_line = node.end_lineno or node.lineno

    code = "\n".join(lines[start_line - 1 : end_line])

    chunk = CodeChunk(
        type="function",
        name=node.name,
        package=package,
        language="python",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        selection_start=start_line,
        selection_end=start_line,
        signature=signature,
        docstring=docstring[:2000],
        code=code,
        metadata={
            "is_async": is_async,
            "visibility": "private" if node.name.startswith("_") else "public",
        },
    )

    return [chunk]


def _extract_class_chunks(
    node: ast.ClassDef, lines: list[str], file_path: str, package: str
) -> list[CodeChunk]:
    """Extract a class and its methods as chunks."""
    chunks = []

    signature = f"class {node.name}"
    if node.bases:
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        signature += f"({bases})"

    docstring = ast.get_docstring(node) or ""

    # Get source code for the class
    start_line = node.lineno
    end_line = node.end_lineno or node.lineno

    code = "\n".join(lines[start_line - 1 : end_line])

    # Add class chunk
    class_chunk = CodeChunk(
        type="class",
        name=node.name,
        package=package,
        language="python",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        selection_start=start_line,
        selection_end=start_line,
        signature=signature,
        docstring=docstring[:2000],
        code=code,
        metadata={"visibility": "private" if node.name.startswith("_") else "public"},
    )
    chunks.append(class_chunk)

    # Extract methods
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(item, ast.AsyncFunctionDef)
            method_signature = _build_signature(item, is_async)

            method_docstring = ast.get_docstring(item) or ""
            method_start = item.lineno
            method_end = item.end_lineno or item.lineno

            method_code = "\n".join(lines[method_start - 1 : method_end])

            method_chunk = CodeChunk(
                type="method",
                name=item.name,
                package=f"{package}.{node.name}",
                language="python",
                file_path=file_path,
                start_line=method_start,
                end_line=method_end,
                selection_start=method_start,
                selection_end=method_start,
                signature=method_signature,
                docstring=method_docstring[:2000],
                code=method_code,
                metadata={
                    "is_async": is_async,
                    "class_name": node.name,
                    "is_static": any(
                        isinstance(dec, ast.Name) and dec.id == "staticmethod"
                        for dec in item.decorator_list
                    ),
                    "visibility": "private"
                    if item.name.startswith("_")
                    else "public",
                },
            )
            chunks.append(method_chunk)

    return chunks


def _build_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool = False
) -> str:
    """Build function signature from AST node."""
    prefix = "async def" if is_async else "def"

    # Get argument string
    args = ast.unparse(node.args)

    # Get return annotation
    return_annotation = ""
    if node.returns:
        return_annotation = f" -> {ast.unparse(node.returns)}"

    return f"{prefix} {node.name}({args}){return_annotation}"


def chunk_generic_file(
    source: str, file_path: str, lang: str, package: str
) -> list[CodeChunk]:
    """
    Extract chunks from source code using regex patterns.

    Supports: JavaScript/TypeScript, Go, C#, and other languages via regex.

    Args:
        source: Source code as string.
        file_path: Absolute path to the file.
        lang: Language identifier.
        package: Package/module name.

    Returns:
        List of CodeChunk objects.
    """
    if lang in ("javascript", "typescript"):
        return _chunk_typescript(source, file_path, lang, package)
    elif lang == "go":
        return _chunk_go(source, file_path, package)
    elif lang == "csharp":
        return _chunk_csharp(source, file_path, package)
    else:
        # Fallback: treat as single file chunk
        return chunk_text_file(source, file_path, lang)


def _chunk_typescript(
    source: str, file_path: str, lang: str, package: str
) -> list[CodeChunk]:
    """Extract functions and classes from JavaScript/TypeScript."""
    chunks = []
    lines = source.split("\n")

    # Extract classes
    class_pattern = r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"
    for match in re.finditer(class_pattern, source):
        class_name = match.group(1)
        # Find line number
        start_pos = match.start()
        line_num = source[:start_pos].count("\n") + 1

        # Extract class body (simple heuristic: find matching braces)
        class_start = match.end()
        brace_count = 0
        class_end = class_start
        in_string = False
        string_char = None

        for i, char in enumerate(source[class_start:], class_start):
            if not in_string:
                if char in ('"', "'", "`"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = i + 1
                        break
            else:
                if char == string_char and (
                    i == 0 or source[i - 1] != "\\"
                ):
                    in_string = False

        class_code = source[match.start() : class_end]
        class_end_line = source[: match.start() + (class_end - match.start())].count(
            "\n"
        )

        signature = f"class {class_name}"
        if match.group(2):
            signature += f" extends {match.group(2)}"

        chunk = CodeChunk(
            type="class",
            name=class_name,
            package=package,
            language=lang,
            file_path=file_path,
            start_line=line_num,
            end_line=class_end_line + 1,
            selection_start=line_num,
            selection_end=line_num,
            signature=signature,
            docstring="",
            code=class_code,
        )
        chunks.append(chunk)

    # Extract functions
    func_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
    for match in re.finditer(func_pattern, source):
        func_name = match.group(1)
        params = match.group(2)

        start_pos = match.start()
        line_num = source[:start_pos].count("\n") + 1

        # Find function body (simple heuristic)
        func_start = match.end()
        brace_count = 0
        func_end = func_start
        in_string = False
        string_char = None

        for i, char in enumerate(source[func_start:], func_start):
            if not in_string:
                if char in ('"', "'", "`"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = i + 1
                        break
            else:
                if char == string_char and (
                    i == 0 or source[i - 1] != "\\"
                ):
                    in_string = False

        func_code = source[match.start() : func_end]
        func_end_line = (
            source[: match.start() + (func_end - match.start())].count("\n") + 1
        )

        is_async = "async" in source[match.start() : match.end()]
        signature = ("async " if is_async else "") + f"function {func_name}({params})"

        chunk = CodeChunk(
            type="function",
            name=func_name,
            package=package,
            language=lang,
            file_path=file_path,
            start_line=line_num,
            end_line=func_end_line,
            selection_start=line_num,
            selection_end=line_num,
            signature=signature,
            docstring="",
            code=func_code,
            metadata={"is_async": is_async},
        )
        chunks.append(chunk)

    return chunks


def _chunk_go(source: str, file_path: str, package: str) -> list[CodeChunk]:
    """Extract functions and types from Go source."""
    chunks = []

    # Extract package name
    pkg_match = re.search(r"package\s+(\w+)", source)
    go_package = pkg_match.group(1) if pkg_match else package

    # Extract functions with optional receiver (methods)
    func_pattern = r"func\s+(?:\(\s*(\w+)\s+(\*?\w+)\s*\)\s+)?(\w+)\s*\(([^)]*)\)\s*(.*)?"
    for match in re.finditer(func_pattern, source):
        receiver_name = match.group(1)
        receiver_type = match.group(2)
        func_name = match.group(3)
        params = match.group(4)
        returns = match.group(5) or ""

        start_pos = match.start()
        line_num = source[:start_pos].count("\n") + 1

        # Find function body
        func_start = match.end()
        brace_count = 0
        func_end = func_start
        in_string = False
        string_char = None

        for i, char in enumerate(source[func_start:], func_start):
            if not in_string:
                if char in ('"', "`"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = i + 1
                        break
            else:
                if char == string_char and (
                    i == 0 or source[i - 1] != "\\"
                ):
                    in_string = False

        func_code = source[match.start() : func_end]
        func_end_line = (
            source[: match.start() + (func_end - match.start())].count("\n") + 1
        )

        # Build signature
        signature = "func"
        if receiver_name and receiver_type:
            signature += f" ({receiver_name} {receiver_type})"
        signature += f" {func_name}({params})"
        if returns.strip():
            signature += f" {returns}"

        is_method = receiver_name is not None
        chunk_type = "method" if is_method else "function"

        chunk = CodeChunk(
            type=chunk_type,
            name=func_name,
            package=go_package if is_method else go_package,
            language="go",
            file_path=file_path,
            start_line=line_num,
            end_line=func_end_line,
            selection_start=line_num,
            selection_end=line_num,
            signature=signature,
            docstring="",
            code=func_code,
            metadata={
                "is_method": is_method,
                "receiver_type": receiver_type if receiver_type else None,
            },
        )
        chunks.append(chunk)

    # Extract type definitions
    type_pattern = r"type\s+(\w+)\s+(struct|interface)"
    for match in re.finditer(type_pattern, source):
        type_name = match.group(1)
        type_kind = match.group(2)

        start_pos = match.start()
        line_num = source[:start_pos].count("\n") + 1

        # Find type body
        type_start = match.end()
        brace_count = 0
        type_end = type_start
        in_string = False
        string_char = None

        for i, char in enumerate(source[type_start:], type_start):
            if not in_string:
                if char in ('"', "`"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        type_end = i + 1
                        break
            else:
                if char == string_char and (
                    i == 0 or source[i - 1] != "\\"
                ):
                    in_string = False

        type_code = source[match.start() : type_end]
        type_end_line = (
            source[: match.start() + (type_end - match.start())].count("\n") + 1
        )

        signature = f"type {type_name} {type_kind}"

        chunk = CodeChunk(
            type="class",
            name=type_name,
            package=go_package,
            language="go",
            file_path=file_path,
            start_line=line_num,
            end_line=type_end_line,
            selection_start=line_num,
            selection_end=line_num,
            signature=signature,
            docstring="",
            code=type_code,
            metadata={"type_kind": type_kind},
        )
        chunks.append(chunk)

    return chunks


def _chunk_csharp(source: str, file_path: str, package: str) -> list[CodeChunk]:
    """Extract classes and methods from C# source."""
    chunks = []

    # Extract namespace
    ns_match = re.search(r"namespace\s+([\w.]+)", source)
    namespace = ns_match.group(1) if ns_match else package

    # Extract class/interface definitions
    class_pattern = r"(?:public|private|protected|internal)?\s*(?:abstract|sealed)?\s*(class|interface|struct)\s+(\w+)"
    for match in re.finditer(class_pattern, source):
        class_kind = match.group(1)
        class_name = match.group(2)

        start_pos = match.start()
        line_num = source[:start_pos].count("\n") + 1

        # Find class body
        class_start = match.end()
        brace_count = 0
        class_end = class_start
        in_string = False
        string_char = None

        for i, char in enumerate(source[class_start:], class_start):
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = i + 1
                        break
            else:
                if char == string_char and (
                    i == 0 or source[i - 1] != "\\"
                ):
                    in_string = False

        class_code = source[match.start() : class_end]
        class_end_line = (
            source[: match.start() + (class_end - match.start())].count("\n") + 1
        )

        signature = f"{class_kind} {class_name}"

        chunk = CodeChunk(
            type="class",
            name=class_name,
            package=namespace,
            language="csharp",
            file_path=file_path,
            start_line=line_num,
            end_line=class_end_line,
            selection_start=line_num,
            selection_end=line_num,
            signature=signature,
            docstring="",
            code=class_code,
            metadata={"class_kind": class_kind},
        )
        chunks.append(chunk)

    # Extract methods
    method_pattern = r"(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)"
    for match in re.finditer(method_pattern, source):
        return_type = match.group(1)
        method_name = match.group(2)
        params = match.group(3)

        start_pos = match.start()
        line_num = source[:start_pos].count("\n") + 1

        # Find method body
        method_start = match.end()
        brace_count = 0
        method_end = method_start
        in_string = False
        string_char = None

        for i, char in enumerate(source[method_start:], method_start):
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        method_end = i + 1
                        break
            else:
                if char == string_char and (
                    i == 0 or source[i - 1] != "\\"
                ):
                    in_string = False

        method_code = source[match.start() : method_end]
        method_end_line = (
            source[: match.start() + (method_end - match.start())].count("\n") + 1
        )

        is_async = "async" in source[match.start() : match.end()]
        is_static = "static" in source[match.start() : match.end()]
        signature = f"{return_type} {method_name}({params})"

        chunk = CodeChunk(
            type="method",
            name=method_name,
            package=namespace,
            language="csharp",
            file_path=file_path,
            start_line=line_num,
            end_line=method_end_line,
            selection_start=line_num,
            selection_end=line_num,
            signature=signature,
            docstring="",
            code=method_code,
            metadata={"is_async": is_async, "is_static": is_static},
        )
        chunks.append(chunk)

    return chunks


def chunk_text_file(source: str, file_path: str, lang: str) -> list[CodeChunk]:
    """
    Create a single chunk for text files (docs, config, etc).

    Args:
        source: File content as string.
        file_path: Absolute path to the file.
        lang: Language/type identifier.

    Returns:
        List with one CodeChunk.
    """
    filename = Path(file_path).name
    file_type = "doc" if lang == "doc" else "config"

    chunk = CodeChunk(
        type=file_type,
        name=filename,
        package="",
        language=lang,
        file_path=file_path,
        start_line=1,
        end_line=source.count("\n") + 1,
        selection_start=1,
        selection_end=1,
        signature=filename,
        docstring="",
        code=source,
    )

    return [chunk]


def chunk_file(file_path: str, config: AppConfig) -> list[CodeChunk]:
    """
    Main routing function: detect language and chunk accordingly.

    Args:
        file_path: Absolute path to the file.
        config: Application configuration.

    Returns:
        List of CodeChunk objects.

    Raises:
        FileNotFoundError: If file does not exist.
        OSError: If file cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file content
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    # Detect language
    lang = detect_language(file_path)

    if lang is None:
        # Unsupported file type
        return []

    # Determine package/module name from file path
    package = _infer_package(file_path)

    # Route to appropriate chunker
    if lang == "python":
        return chunk_python_file(source, file_path, package)
    elif lang in ("javascript", "typescript", "go", "csharp", "rust", "java", "ruby", "php", "cpp", "swift", "kotlin"):
        return chunk_generic_file(source, file_path, lang, package)
    else:
        # Config, doc, or other text files
        return chunk_text_file(source, file_path, lang)


def _infer_package(file_path: str) -> str:
    """
    Infer package/module name from file path.

    Args:
        file_path: Absolute file path.

    Returns:
        Package name (e.g., "mcp_code_rag.chunker" for mcp_code_rag/chunker.py).
    """
    path = Path(file_path)

    # For Python files, use directory structure
    parts = []
    current = path.parent

    # Walk up until we hit a directory without __init__.py or reach root
    while current.is_dir():
        if not (current / "__init__.py").exists():
            break
        parts.insert(0, current.name)
        current = current.parent

    # Remove .py extension and add filename
    if path.suffix == ".py":
        parts.append(path.stem)

    return ".".join(parts) if parts else path.stem
