"""Tests for symbol extraction and language detection."""

import tempfile
from pathlib import Path

import pytest

from mcp_code_rag.extractors import (
    ClassDescriptor,
    FunctionDescriptor,
    SymbolLocation,
    detect_language,
    extract_imports_generic,
    extract_imports_python,
    is_excluded,
    resolve_import_to_path,
)


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_language_python(self):
        """Test detecting Python files."""
        assert detect_language("script.py") == "python"
        assert detect_language("/path/to/file.py") == "python"

    def test_detect_language_go(self):
        """Test detecting Go files."""
        assert detect_language("main.go") == "go"
        assert detect_language("/path/to/module.go") == "go"

    def test_detect_language_yaml(self):
        """Test detecting YAML config files."""
        assert detect_language("config.yaml") == "config"
        assert detect_language("settings.yml") == "config"

    def test_detect_language_lock_file_returns_none(self):
        """Test that lock files return None."""
        assert detect_language("package-lock.json") is None
        assert detect_language("Gemfile.lock") is None
        assert detect_language("Cargo.lock") is None

    def test_detect_language_minified_js_returns_none(self):
        """Test that minified JavaScript returns None."""
        assert detect_language("bundle.min.js") is None
        assert detect_language("app.min.css") is None

    def test_detect_language_ds_store_returns_none(self):
        """Test that .DS_Store returns None."""
        assert detect_language(".DS_Store") is None

    def test_detect_language_javascript(self):
        """Test detecting JavaScript files."""
        assert detect_language("app.js") == "javascript"
        assert detect_language("module.mjs") == "javascript"

    def test_detect_language_typescript(self):
        """Test detecting TypeScript files."""
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "typescript"

    def test_detect_language_csharp(self):
        """Test detecting C# files."""
        assert detect_language("Program.cs") == "csharp"

    def test_detect_language_rust(self):
        """Test detecting Rust files."""
        assert detect_language("main.rs") == "rust"

    def test_detect_language_java(self):
        """Test detecting Java files."""
        assert detect_language("Main.java") == "java"

    def test_detect_language_ruby(self):
        """Test detecting Ruby files."""
        assert detect_language("script.rb") == "ruby"

    def test_detect_language_php(self):
        """Test detecting PHP files."""
        assert detect_language("index.php") == "php"

    def test_detect_language_cpp(self):
        """Test detecting C/C++ files."""
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("header.h") == "cpp"
        assert detect_language("file.c") == "cpp"

    def test_detect_language_swift(self):
        """Test detecting Swift files."""
        assert detect_language("app.swift") == "swift"

    def test_detect_language_kotlin(self):
        """Test detecting Kotlin files."""
        assert detect_language("main.kt") == "kotlin"

    def test_detect_language_documentation(self):
        """Test detecting documentation files."""
        assert detect_language("README.md") == "doc"
        assert detect_language("guide.markdown") == "doc"
        assert detect_language("notes.txt") == "doc"

    def test_detect_language_config_formats(self):
        """Test detecting various config formats."""
        assert detect_language("pyproject.toml") == "config"
        assert detect_language("config.json") == "config"
        assert detect_language("app.env") == "config"
        assert detect_language("web.xml") == "config"

    def test_detect_language_unknown_returns_none(self):
        """Test that unknown extensions return None."""
        assert detect_language("file.unknown") is None
        assert detect_language("data.xyz") is None


class TestIsExcluded:
    """Tests for is_excluded function."""

    def test_is_excluded_node_modules(self):
        """Test excluding node_modules directory."""
        patterns = ["node_modules"]
        assert is_excluded("node_modules", patterns) is True
        assert is_excluded("path/node_modules/package", patterns) is True
        assert is_excluded("/project/node_modules/react", patterns) is True

    def test_is_excluded_git_directory(self):
        """Test excluding .git directory."""
        patterns = [".git"]
        assert is_excluded(".git", patterns) is True
        assert is_excluded(".git/objects", patterns) is True

    def test_is_excluded_pycache(self):
        """Test excluding __pycache__ directory."""
        patterns = ["__pycache__"]
        assert is_excluded("__pycache__", patterns) is True
        assert is_excluded("src/__pycache__", patterns) is True

    def test_is_excluded_wildcards(self):
        """Test wildcard patterns."""
        patterns = ["*.min.js", "*.lock"]
        assert is_excluded("bundle.min.js", patterns) is True
        assert is_excluded("package-lock.json", patterns) is True
        assert is_excluded("normal.js", patterns) is False

    def test_is_excluded_multiple_patterns(self):
        """Test matching against multiple patterns."""
        patterns = ["node_modules", ".git", "__pycache__", "*.min.js"]
        assert is_excluded("node_modules", patterns) is True
        assert is_excluded(".git", patterns) is True
        assert is_excluded("app.min.js", patterns) is True
        assert is_excluded("app.js", patterns) is False

    def test_is_excluded_no_match_returns_false(self):
        """Test that non-matching paths return False."""
        patterns = ["node_modules", ".git"]
        assert is_excluded("src/main.py", patterns) is False
        assert is_excluded("lib/utils.js", patterns) is False

    def test_is_excluded_empty_patterns(self):
        """Test that empty pattern list excludes nothing."""
        patterns = []
        assert is_excluded("node_modules", patterns) is False
        assert is_excluded("anything.py", patterns) is False

    def test_is_excluded_directory_paths(self):
        """Test excluding directory patterns in path."""
        patterns = ["venv", ".venv", "dist", "build"]
        assert is_excluded("project/venv/lib", patterns) is True
        assert is_excluded(".venv/bin", patterns) is True
        assert is_excluded("dist/bundle.js", patterns) is True

    def test_is_excluded_complex_patterns(self):
        """Test complex fnmatch patterns."""
        patterns = ["*.egg-info", ".mypy_cache", ".ruff_cache"]
        assert is_excluded("package.egg-info", patterns) is True
        assert is_excluded(".mypy_cache/file", patterns) is True
        assert is_excluded(".ruff_cache", patterns) is True


class TestExtractImportsPython:
    """Tests for extract_imports_python function."""

    def test_extract_imports_simple_import(self):
        """Test extracting simple import statements."""
        source = "import os"
        imports = extract_imports_python(source)
        assert "os" in imports

    def test_extract_imports_from_import(self):
        """Test extracting from...import statements."""
        source = "from pathlib import Path"
        imports = extract_imports_python(source)
        assert "pathlib" in imports

    def test_extract_imports_multiple_modules(self):
        """Test extracting multiple import statements."""
        source = """
import os
import sys
from pathlib import Path
from collections import defaultdict
"""
        imports = extract_imports_python(source)
        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports
        assert "collections" in imports

    def test_extract_imports_nested_modules(self):
        """Test extracting nested module imports."""
        source = "from django.contrib.auth.models import User"
        imports = extract_imports_python(source)
        assert "django.contrib.auth.models" in imports

    def test_extract_imports_multiple_from_import(self):
        """Test from...import with multiple items."""
        source = "from module import func1, func2, func3"
        imports = extract_imports_python(source)
        assert "module" in imports

    def test_extract_imports_aliased_imports(self):
        """Test extracting aliased imports."""
        source = """
import numpy as np
from pandas import DataFrame as DF
"""
        imports = extract_imports_python(source)
        assert "numpy" in imports
        assert "pandas" in imports

    def test_extract_imports_syntax_error_returns_empty(self):
        """Test that syntax errors return empty list."""
        source = "this is not valid python :("
        imports = extract_imports_python(source)
        assert imports == []

    def test_extract_imports_no_imports_returns_empty(self):
        """Test code with no imports returns empty list."""
        source = """
def hello():
    print("world")

x = 42
"""
        imports = extract_imports_python(source)
        assert imports == []

    def test_extract_imports_conditional_imports(self):
        """Test extracting conditionally imported modules."""
        source = """
import sys
if sys.platform == "win32":
    import winreg
else:
    import pwd
"""
        imports = extract_imports_python(source)
        assert "sys" in imports
        assert "winreg" in imports
        assert "pwd" in imports


class TestExtractImportsGeneric:
    """Tests for extract_imports_generic function."""

    def test_extract_imports_javascript_import_from(self):
        """Test extracting JavaScript import...from statements."""
        source = 'import React from "react"'
        imports = extract_imports_generic(source, "javascript")
        assert "react" in imports

    def test_extract_imports_javascript_default_import(self):
        """Test extracting JavaScript default imports."""
        source = 'import lodash from "lodash"'
        imports = extract_imports_generic(source, "javascript")
        assert "lodash" in imports

    def test_extract_imports_javascript_side_effect_import(self):
        """Test extracting JavaScript side-effect imports."""
        source = 'import "polyfill.js"'
        imports = extract_imports_generic(source, "javascript")
        assert "polyfill.js" in imports

    def test_extract_imports_javascript_multiple(self):
        """Test extracting multiple JavaScript imports."""
        source = """
import React from "react"
import { useState } from "react"
import lodash from "lodash"
import "./styles.css"
"""
        imports = extract_imports_generic(source, "javascript")
        assert "react" in imports
        assert "lodash" in imports
        assert "./styles.css" in imports

    def test_extract_imports_typescript_same_as_javascript(self):
        """Test TypeScript imports work like JavaScript."""
        source = 'import { Component } from "ng/core"'
        imports = extract_imports_generic(source, "typescript")
        assert "ng/core" in imports

    def test_extract_imports_go_single_import(self):
        """Test extracting Go import statements."""
        source = 'import "fmt"'
        imports = extract_imports_generic(source, "go")
        assert "fmt" in imports

    def test_extract_imports_go_grouped_imports(self):
        """Test extracting Go grouped imports."""
        source = '''
import (
    "fmt"
    "os"
    "path/filepath"
)
'''
        imports = extract_imports_generic(source, "go")
        assert "fmt" in imports
        assert "os" in imports
        assert "path/filepath" in imports

    def test_extract_imports_csharp_using(self):
        """Test extracting C# using statements."""
        source = """
using System;
using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
"""
        imports = extract_imports_generic(source, "csharp")
        assert "System" in imports
        assert "System.Collections.Generic" in imports
        assert "Microsoft.AspNetCore.Mvc" in imports

    def test_extract_imports_no_imports_returns_empty(self):
        """Test that code with no imports returns empty list."""
        source = """
function hello() {
    console.log("world");
}
"""
        imports = extract_imports_generic(source, "javascript")
        assert imports == []


class TestSymbolLocation:
    """Tests for SymbolLocation dataclass."""

    def test_symbol_location_creation(self):
        """Test creating SymbolLocation instance."""
        loc = SymbolLocation(
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
        )
        assert loc.file_path == "/path/to/file.py"
        assert loc.start_line == 10
        assert loc.end_line == 20


class TestFunctionDescriptor:
    """Tests for FunctionDescriptor dataclass."""

    def test_function_descriptor_minimal(self):
        """Test creating minimal FunctionDescriptor."""
        func = FunctionDescriptor(
            language="python",
            kind="function",
            name="my_function",
        )
        assert func.language == "python"
        assert func.kind == "function"
        assert func.name == "my_function"
        assert func.parameters == []
        assert func.is_static is False
        assert func.is_async is False

    def test_function_descriptor_full(self):
        """Test creating fully populated FunctionDescriptor."""
        loc = SymbolLocation(
            file_path="/path/to/file.py",
            start_line=10,
            end_line=15,
        )
        func = FunctionDescriptor(
            language="python",
            kind="method",
            name="my_method",
            namespace="MyClass",
            signature="def my_method(self, x: int) -> str",
            description="My method docstring",
            location=loc,
            parameters=[{"name": "x", "type": "int"}],
            returns=[{"type": "str"}],
            visibility="public",
            is_static=False,
            is_async=True,
            code="def my_method(self, x: int) -> str:\n    return str(x)",
            tags=["method", "public"],
        )
        assert func.name == "my_method"
        assert func.namespace == "MyClass"
        assert func.is_async is True
        assert len(func.parameters) == 1


class TestClassDescriptor:
    """Tests for ClassDescriptor dataclass."""

    def test_class_descriptor_minimal(self):
        """Test creating minimal ClassDescriptor."""
        cls = ClassDescriptor(
            language="python",
            kind="class",
            name="MyClass",
        )
        assert cls.language == "python"
        assert cls.kind == "class"
        assert cls.name == "MyClass"
        assert cls.fields == []
        assert cls.methods == []
        assert cls.base_classes == []

    def test_class_descriptor_with_methods(self):
        """Test ClassDescriptor with methods."""
        method = FunctionDescriptor(
            language="python",
            kind="method",
            name="get_name",
        )
        cls = ClassDescriptor(
            language="python",
            kind="class",
            name="Person",
            full_name="models.Person",
            methods=[method],
            base_classes=["BaseModel"],
        )
        assert len(cls.methods) == 1
        assert cls.methods[0].name == "get_name"
        assert "BaseModel" in cls.base_classes


class TestResolveImportToPython:
    """Tests for resolve_import_to_path with Python imports."""

    def test_resolve_python_absolute_import(self):
        """Test resolving absolute Python imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create a module
            (tmpdir_path / "mymodule.py").touch()

            source_file = str(tmpdir_path / "main.py")
            result = resolve_import_to_path(
                "mymodule",
                source_file,
                str(tmpdir_path),
                "python",
            )
            assert result is not None
            assert "mymodule.py" in result

    def test_resolve_python_relative_import(self):
        """Test resolving relative Python imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()
            (src_dir / "__init__.py").touch()
            (src_dir / "utils.py").touch()

            source_file = str(src_dir / "main.py")
            result = resolve_import_to_path(
                ".utils",
                source_file,
                str(tmpdir_path),
                "python",
            )
            assert result is not None
            assert "utils.py" in result

    def test_resolve_python_package_import(self):
        """Test resolving Python package imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pkg_dir = tmpdir_path / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").touch()

            source_file = str(tmpdir_path / "main.py")
            result = resolve_import_to_path(
                "mypackage",
                source_file,
                str(tmpdir_path),
                "python",
            )
            assert result is not None
            assert "mypackage" in result


class TestResolveImportToPathJS:
    """Tests for resolve_import_to_path with JavaScript imports."""

    def test_resolve_js_relative_import(self):
        """Test resolving relative JavaScript imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "utils.js").touch()

            source_file = str(tmpdir_path / "main.js")
            result = resolve_import_to_path(
                "./utils.js",
                source_file,
                str(tmpdir_path),
                "javascript",
            )
            assert result is not None
            assert "utils.js" in result

    def test_resolve_js_index_file(self):
        """Test resolving a directory import to its index.js file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            utils_dir = tmpdir_path / "utils"
            utils_dir.mkdir()
            (utils_dir / "index.js").touch()

            source_file = str(tmpdir_path / "main.js")
            result = resolve_import_to_path(
                "./utils",
                source_file,
                str(tmpdir_path),
                "javascript",
            )
            assert result is not None
            assert "index.js" in result

    def test_resolve_js_node_modules_not_found(self):
        """Test that non-relative import to missing node_modules returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = str(Path(tmpdir) / "main.js")
            result = resolve_import_to_path(
                "some-unknown-package",
                source_file,
                tmpdir,
                "javascript",
            )
            assert result is None

    def test_resolve_unsupported_language_returns_none(self):
        """Test that unsupported languages return None from resolve_import_to_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_import_to_path(
                "somemodule",
                str(Path(tmpdir) / "file.rb"),
                tmpdir,
                "ruby",
            )
            assert result is None


class TestResolveImportToGo:
    """Tests for resolve_import_to_path with Go imports."""

    def test_resolve_go_import_main_go(self):
        """Test resolving a Go import when the target directory has main.go."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mod_dir = tmpdir_path / "mymod"
            mod_dir.mkdir()
            (mod_dir / "main.go").touch()

            source_file = str(tmpdir_path / "app.go")
            result = resolve_import_to_path(
                "mymod",
                source_file,
                str(tmpdir_path),
                "go",
            )
            assert result is not None
            assert "main.go" in result

    def test_resolve_go_import_directory_with_go_file(self):
        """Test resolving a Go import to the first .go file in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            utils_dir = tmpdir_path / "internal" / "utils"
            utils_dir.mkdir(parents=True)
            (utils_dir / "helper.go").touch()

            source_file = str(tmpdir_path / "main.go")
            result = resolve_import_to_path(
                "internal/utils",
                source_file,
                str(tmpdir_path),
                "go",
            )
            assert result is not None
            assert "helper.go" in result

    def test_resolve_go_import_not_found(self):
        """Test that a missing Go module returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_import_to_path(
                "nonexistent/module",
                str(Path(tmpdir) / "main.go"),
                tmpdir,
                "go",
            )
            assert result is None


class TestResolveImportToCSharp:
    """Tests for resolve_import_to_path with C# imports."""

    def test_resolve_csharp_import_found(self):
        """Test resolving a C# using statement to a file that declares the namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cs_file = tmpdir_path / "Services.cs"
            cs_file.write_text("namespace MyApp.Services\n{\n    // ...\n}\n")

            result = resolve_import_to_path(
                "MyApp.Services",
                str(tmpdir_path / "Main.cs"),
                str(tmpdir_path),
                "csharp",
            )
            assert result is not None
            assert "Services.cs" in result

    def test_resolve_csharp_import_not_found(self):
        """Test that a C# namespace with no matching file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_import_to_path(
                "NonExistent.Namespace",
                str(Path(tmpdir) / "Main.cs"),
                tmpdir,
                "csharp",
            )
            assert result is None
