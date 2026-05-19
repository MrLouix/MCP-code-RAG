"""Tests for AST-based Python code chunking."""

import pytest
from mcp_code_rag.chunker import CodeChunk, chunk_python_file


def test_chunk_simple_function():
    """Test chunking a simple function."""
    source = '''
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
'''

    chunks = chunk_python_file(source, "/test/module.py", "test_module")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "function"
    assert chunk.name == "greet"
    assert chunk.language == "python"
    assert "def greet" in chunk.signature
    assert "Greet someone" in chunk.docstring
    assert "return f" in chunk.code


def test_chunk_async_function():
    """Test chunking an async function."""
    source = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    return {"data": "result"}
'''

    chunks = chunk_python_file(source, "/test/async.py", "test_module")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "function"
    assert chunk.name == "fetch_data"
    assert "async def" in chunk.signature
    assert chunk.metadata["is_async"] is True


def test_chunk_class_with_methods():
    """Test chunking a class with methods."""
    source = '''
class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract two numbers."""
        return a - b
'''

    chunks = chunk_python_file(source, "/test/calc.py", "test_module")

    assert len(chunks) == 3  # Class + 2 methods

    # Check class chunk
    class_chunk = chunks[0]
    assert class_chunk.type == "class"
    assert class_chunk.name == "Calculator"

    # Check methods
    add_chunk = chunks[1]
    assert add_chunk.type == "method"
    assert add_chunk.name == "add"
    assert add_chunk.metadata["class_name"] == "Calculator"

    subtract_chunk = chunks[2]
    assert subtract_chunk.type == "method"
    assert subtract_chunk.name == "subtract"


def test_chunk_class_with_private_methods():
    """Test that private methods are marked correctly."""
    source = '''
class Service:
    """Service class."""

    def _private_method(self):
        """Private helper."""
        pass

    def public_method(self):
        """Public method."""
        pass
'''

    chunks = chunk_python_file(source, "/test/service.py", "test_module")

    private_chunk = chunks[1]
    assert private_chunk.metadata["visibility"] == "private"

    public_chunk = chunks[2]
    assert public_chunk.metadata["visibility"] == "public"


def test_chunk_class_with_inheritance():
    """Test chunking a class with base classes."""
    source = '''
class User(BaseModel):
    """User model."""
    name: str
    email: str
'''

    chunks = chunk_python_file(source, "/test/models.py", "test_module")

    assert len(chunks) >= 1
    class_chunk = chunks[0]
    assert class_chunk.type == "class"
    assert "BaseModel" in class_chunk.signature


def test_chunk_static_method():
    """Test chunking a static method."""
    source = '''
class Utils:
    @staticmethod
    def format_date(date_str: str) -> str:
        """Format date string."""
        return date_str.strip()
'''

    chunks = chunk_python_file(source, "/test/utils.py", "test_module")

    # Find the static method chunk
    static_chunk = None
    for chunk in chunks:
        if chunk.name == "format_date":
            static_chunk = chunk
            break

    assert static_chunk is not None
    assert static_chunk.type == "method"
    assert static_chunk.metadata["is_static"] is True


def test_chunk_with_syntax_error():
    """Test that syntax errors are handled gracefully."""
    source = '''
def broken(:
    pass
'''

    chunks = chunk_python_file(source, "/test/broken.py", "test_module")

    # Should return empty list, not raise exception
    assert chunks == []


def test_chunk_multiline_docstring():
    """Test that docstrings are extracted correctly."""
    source = '''
def process_data(data: dict) -> dict:
    """
    Process input data.

    This is a longer description
    with multiple lines.

    Args:
        data: Input dictionary

    Returns:
        Processed dictionary
    """
    return data
'''

    chunks = chunk_python_file(source, "/test/processor.py", "test_module")

    chunk = chunks[0]
    assert "Process input data" in chunk.docstring
    assert "Args:" in chunk.docstring


def test_chunk_respects_max_docstring_length():
    """Test that docstrings are truncated to 2000 chars."""
    source = '''
def long_doc():
    """{}"""
    pass
'''.format("x" * 3000)

    chunks = chunk_python_file(source, "/test/long.py", "test_module")

    chunk = chunks[0]
    assert len(chunk.docstring) <= 2000


def test_chunk_function_with_decorator():
    """Test that decorated functions are still extracted as chunks."""
    source = '''
import functools

@app.route("/home")
def index():
    """Home page route."""
    return "hello"

@functools.lru_cache(maxsize=128)
@some_other_decorator
def cached_lookup(key: str) -> str:
    """Cached lookup function."""
    return key
'''

    chunks = chunk_python_file(source, "/test/views.py", "test_module")

    assert len(chunks) == 2
    names = {c.name for c in chunks}
    assert "index" in names
    assert "cached_lookup" in names

    # Decorators don't change chunk type
    for chunk in chunks:
        assert chunk.type == "function"


def test_chunk_class_with_nested_class():
    """Test that nested classes in a class body are not separately extracted."""
    source = '''
class Outer:
    """Outer class."""

    class Inner:
        """Inner nested class — not extracted separately by the AST chunker."""
        value: int = 0

    def method(self):
        """A method on the outer class."""
        pass

    async def async_method(self):
        """An async method."""
        pass
'''

    chunks = chunk_python_file(source, "/test/nested.py", "test_module")

    # The chunker extracts: Outer class + method + async_method methods
    # Inner class is in Outer.body but is a ClassDef, not FunctionDef, so it
    # is NOT extracted as a separate chunk.
    names = {c.name for c in chunks}
    assert "Outer" in names
    assert "method" in names
    assert "async_method" in names
    assert "Inner" not in names  # nested class not separately chunked

    outer = next(c for c in chunks if c.name == "Outer")
    assert outer.type == "class"

    method = next(c for c in chunks if c.name == "method")
    assert method.type == "method"
    assert method.metadata["class_name"] == "Outer"

    async_method = next(c for c in chunks if c.name == "async_method")
    assert async_method.metadata["is_async"] is True
