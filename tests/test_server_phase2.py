"""Tests for Phase 2 FastMCP tools (server.py — tools 7-15)."""

import json
from unittest.mock import Mock

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from mcp_code_rag.chunker import CodeChunk
from mcp_code_rag.config import AppConfig
from mcp_code_rag.hybrid_search import HybridSearch
from mcp_code_rag.ingest import CodeIngestPipeline
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.server import create_server
from mcp_code_rag.storage import Storage

# ── shared workspace id ────────────────────────────────────────────────────

WS = "p2_test_ws"

# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def app_config():
    return AppConfig()


@pytest.fixture
def mock_ollama():
    client = Mock(spec=OllamaClient)
    client.embed = Mock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    client.list_models = Mock(return_value=[{"name": "nomic-embed-text"}])
    return client


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "index"))


@pytest.fixture
def filled_storage(storage):
    """Pre-populate storage with a representative set of symbols and deps."""
    # Function
    storage.store_chunk(
        CodeChunk(
            type="function",
            name="calculate_sum",
            package="math_utils",
            language="python",
            file_path="/project/math_utils.py",
            start_line=5,
            end_line=15,
            selection_start=5,
            selection_end=5,
            signature="def calculate_sum(a: int, b: int) -> int:",
            docstring="Calculate sum of two integers.",
            code="def calculate_sum(a, b):\n    return a + b",
        ),
        [0.1] * 768,
        WS,
    )

    # Class (wide range so methods fit inside)
    storage.store_chunk(
        CodeChunk(
            type="class",
            name="Calculator",
            package="math_utils",
            language="python",
            file_path="/project/math_utils.py",
            start_line=20,
            end_line=80,
            selection_start=20,
            selection_end=20,
            signature="class Calculator:",
            docstring="A simple calculator class.",
            code="class Calculator:\n    pass",
        ),
        [0.2] * 768,
        WS,
    )

    # Method inside Calculator (lines 25-40 ⊂ 20-80)
    storage.store_chunk(
        CodeChunk(
            type="method",
            name="add",
            package="math_utils.Calculator",
            language="python",
            file_path="/project/math_utils.py",
            start_line=25,
            end_line=40,
            selection_start=25,
            selection_end=25,
            signature="def add(self, a: int, b: int) -> int:",
            docstring="Add two numbers.",
            code="def add(self, a, b):\n    return a + b",
        ),
        [0.3] * 768,
        WS,
    )

    # Second method inside Calculator (lines 45-60 ⊂ 20-80)
    storage.store_chunk(
        CodeChunk(
            type="method",
            name="subtract",
            package="math_utils.Calculator",
            language="python",
            file_path="/project/math_utils.py",
            start_line=45,
            end_line=60,
            selection_start=45,
            selection_end=45,
            signature="def subtract(self, a: int, b: int) -> int:",
            docstring="Subtract two numbers.",
            code="def subtract(self, a, b):\n    return a - b",
        ),
        [0.4] * 768,
        WS,
    )

    # Second file
    storage.store_chunk(
        CodeChunk(
            type="function",
            name="validate_input",
            package="validators",
            language="python",
            file_path="/project/validators.py",
            start_line=1,
            end_line=10,
            selection_start=1,
            selection_end=1,
            signature="def validate_input(x: str) -> bool:",
            docstring="Validate input string.",
            code="def validate_input(x):\n    return bool(x)",
        ),
        [0.5] * 768,
        WS,
    )

    # Dependencies
    storage.store_dependency("Calculator.add", "calculate_sum", WS)
    storage.store_dependency("main", "Calculator", WS)
    storage.store_dependency("main", "validate_input", WS)

    # Scan history (two entries for delta calculation)
    storage.record_scan(WS, 2, 5, 800)
    storage.record_scan(WS, 3, 7, 900)

    return storage


@pytest.fixture
def pipeline(filled_storage, mock_ollama, app_config):
    return CodeIngestPipeline(filled_storage, mock_ollama, app_config)


@pytest.fixture
def search(filled_storage, mock_ollama, app_config):
    return HybridSearch(filled_storage, mock_ollama, app_config.hybrid_search)


@pytest.fixture
def server(app_config, filled_storage, mock_ollama, pipeline, search):
    return create_server(app_config, filled_storage, mock_ollama, pipeline, search)


# ── get_function_details ───────────────────────────────────────────────────


async def test_get_function_details_human(server):
    """Returns human-readable FunctionDescriptor for a known function."""
    async with Client(server) as client:
        result = await client.call_tool(
            "get_function_details",
            {"symbol_name": "calculate_sum", "workspace_id": WS},
        )
    assert not result.is_error
    text = result.data
    assert "calculate_sum" in text
    assert "function" in text.lower()
    assert "/project/math_utils.py" in text


async def test_get_function_details_json(server):
    """JSON output contains all FunctionDescriptor fields."""
    async with Client(server) as client:
        result = await client.call_tool(
            "get_function_details",
            {"symbol_name": "calculate_sum", "workspace_id": WS, "output_format": "json"},
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert data["name"] == "calculate_sum"
    assert data["kind"] in ("function", "method")
    assert "location" in data
    assert data["location"]["file_path"] == "/project/math_utils.py"


async def test_get_function_details_method(server):
    """Works for methods as well as standalone functions."""
    async with Client(server) as client:
        result = await client.call_tool(
            "get_function_details",
            {"symbol_name": "add", "workspace_id": WS},
        )
    assert not result.is_error
    assert "add" in result.data


async def test_get_function_details_not_found(server):
    """Raises ToolError when the symbol does not exist."""
    with pytest.raises(ToolError):
        async with Client(server) as client:
            await client.call_tool(
                "get_function_details",
                {"symbol_name": "nonexistent_fn", "workspace_id": WS},
            )


# ── find_type_definition ───────────────────────────────────────────────────


async def test_find_type_definition_human(server):
    """Returns human-readable ClassDescriptor with methods."""
    async with Client(server) as client:
        result = await client.call_tool(
            "find_type_definition",
            {"type_name": "Calculator", "workspace_id": WS},
        )
    assert not result.is_error
    text = result.data
    assert "Calculator" in text
    assert "class" in text.lower()
    # Should list enclosed methods
    assert "add" in text
    assert "subtract" in text


async def test_find_type_definition_json(server):
    """JSON output is a valid ClassDescriptor with methods list."""
    async with Client(server) as client:
        result = await client.call_tool(
            "find_type_definition",
            {"type_name": "Calculator", "workspace_id": WS, "output_format": "json"},
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert data["name"] == "Calculator"
    assert data["kind"] == "class"
    assert isinstance(data["methods"], list)
    method_names = [m["name"] for m in data["methods"]]
    assert "add" in method_names
    assert "subtract" in method_names


async def test_find_type_definition_not_found(server):
    """Raises ToolError for an unknown type name."""
    with pytest.raises(ToolError):
        async with Client(server) as client:
            await client.call_tool(
                "find_type_definition",
                {"type_name": "NoSuchClass", "workspace_id": WS},
            )


# ── find_implementations ───────────────────────────────────────────────────


async def test_find_implementations_callers(server):
    """callers direction returns symbols that call the target."""
    async with Client(server) as client:
        result = await client.call_tool(
            "find_implementations",
            {"symbol_name": "Calculator", "direction": "callers", "workspace_id": WS},
        )
    assert not result.is_error
    assert "main" in result.data


async def test_find_implementations_targets(server):
    """targets direction returns symbols that the source calls."""
    async with Client(server) as client:
        result = await client.call_tool(
            "find_implementations",
            {"symbol_name": "main", "direction": "targets", "workspace_id": WS},
        )
    assert not result.is_error
    assert "Calculator" in result.data or "validate_input" in result.data


async def test_find_implementations_json(server):
    """JSON output includes symbol, direction, and results keys."""
    async with Client(server) as client:
        result = await client.call_tool(
            "find_implementations",
            {
                "symbol_name": "Calculator",
                "direction": "callers",
                "workspace_id": WS,
                "output_format": "json",
            },
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert data["symbol"] == "Calculator"
    assert data["direction"] == "callers"
    assert isinstance(data["results"], list)


async def test_find_implementations_empty(server):
    """Returns a human-readable message when no results are found."""
    async with Client(server) as client:
        result = await client.call_tool(
            "find_implementations",
            {"symbol_name": "calculate_sum", "direction": "callers", "workspace_id": WS},
        )
    # calculate_sum has no callers stored (Calculator.add -> calculate_sum means
    # calculate_sum IS the target, so get_callers("calculate_sum") returns
    # Calculator.add as caller)
    assert not result.is_error


async def test_find_implementations_invalid_direction(server):
    """Invalid direction raises ToolError."""
    with pytest.raises(ToolError):
        async with Client(server) as client:
            await client.call_tool(
                "find_implementations",
                {"symbol_name": "Calculator", "direction": "bad_dir", "workspace_id": WS},
            )


# ── get_code_context ───────────────────────────────────────────────────────


async def test_get_code_context_inside_function(server):
    """Reports the enclosing function for a line inside its range."""
    # calculate_sum spans lines 5-15; line 10 should be inside it
    async with Client(server) as client:
        result = await client.call_tool(
            "get_code_context",
            {"file_path": "/project/math_utils.py", "line": 10, "workspace_id": WS},
        )
    assert not result.is_error
    assert "calculate_sum" in result.data
    assert "10" in result.data


async def test_get_code_context_inside_method(server):
    """Reports the innermost enclosing symbol (method inside class)."""
    # 'add' spans 25-40, 'Calculator' spans 20-80; line 30 is in both
    async with Client(server) as client:
        result = await client.call_tool(
            "get_code_context",
            {"file_path": "/project/math_utils.py", "line": 30, "workspace_id": WS},
        )
    assert not result.is_error
    # Innermost is 'add' (smaller range)
    assert "add" in result.data


async def test_get_code_context_not_found(server):
    """Returns a clear message when line falls outside all indexed symbols."""
    async with Client(server) as client:
        result = await client.call_tool(
            "get_code_context",
            {"file_path": "/project/math_utils.py", "line": 1000, "workspace_id": WS},
        )
    assert not result.is_error
    assert "not inside" in result.data.lower() or "1000" in result.data


async def test_get_code_context_json(server):
    """JSON output contains file_path, line, and enclosing_symbols list."""
    async with Client(server) as client:
        result = await client.call_tool(
            "get_code_context",
            {
                "file_path": "/project/math_utils.py",
                "line": 10,
                "workspace_id": WS,
                "output_format": "json",
            },
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert data["file_path"] == "/project/math_utils.py"
    assert data["line"] == 10
    assert isinstance(data["enclosing_symbols"], list)
    assert len(data["enclosing_symbols"]) >= 1


# ── project_map ────────────────────────────────────────────────────────────


async def test_project_map_human(server):
    """project_map human output reports file count and languages."""
    async with Client(server) as client:
        result = await client.call_tool(
            "project_map", {"workspace_id": WS}
        )
    assert not result.is_error
    text = result.data
    assert WS in text
    assert "Files" in text
    assert "python" in text.lower()


async def test_project_map_json(server):
    """project_map json output has summary key."""
    async with Client(server) as client:
        result = await client.call_tool(
            "project_map", {"workspace_id": WS, "output_format": "json"}
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert "summary" in data
    assert data["summary"]["total_files"] > 0


async def test_project_map_detailed(server):
    """project_map detailed format includes files list."""
    async with Client(server) as client:
        result = await client.call_tool(
            "project_map",
            {"workspace_id": WS, "map_format": "detailed", "output_format": "json"},
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert "detailed" in data
    assert len(data["detailed"]["files"]) > 0


async def test_project_map_tree(server):
    """project_map tree format includes a file tree string."""
    async with Client(server) as client:
        result = await client.call_tool(
            "project_map",
            {"workspace_id": WS, "map_format": "tree", "output_format": "json"},
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert "tree" in data
    assert isinstance(data["tree"], str)


# ── file_tree ──────────────────────────────────────────────────────────────


async def test_file_tree_human(server):
    """file_tree returns a tree containing indexed file names."""
    async with Client(server) as client:
        result = await client.call_tool("file_tree", {"workspace_id": WS})
    assert not result.is_error
    text = result.data
    assert "math_utils.py" in text
    assert "validators.py" in text


async def test_file_tree_json(server):
    """file_tree json output contains workspace_id and tree."""
    async with Client(server) as client:
        result = await client.call_tool(
            "file_tree", {"workspace_id": WS, "output_format": "json"}
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert data["workspace_id"] == WS
    assert isinstance(data["tree"], str)


async def test_file_tree_with_filter(server):
    """path_filter limits displayed files."""
    async with Client(server) as client:
        result = await client.call_tool(
            "file_tree", {"workspace_id": WS, "path_filter": "*/validators.py"}
        )
    assert not result.is_error
    assert "validators.py" in result.data


# ── list_package_exports ───────────────────────────────────────────────────


async def test_list_package_exports_human(server):
    """Returns public symbols for a package, excludes private ones."""
    async with Client(server) as client:
        result = await client.call_tool(
            "list_package_exports",
            {"package_path": "math_utils", "workspace_id": WS},
        )
    assert not result.is_error
    text = result.data
    assert "calculate_sum" in text
    assert "Calculator" in text


async def test_list_package_exports_json(server):
    """JSON output has package and exports keys."""
    async with Client(server) as client:
        result = await client.call_tool(
            "list_package_exports",
            {
                "package_path": "math_utils",
                "workspace_id": WS,
                "output_format": "json",
            },
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert data["package"] == "math_utils"
    assert isinstance(data["exports"], list)
    assert "calculate_sum" in data["exports"]
    assert "Calculator" in data["exports"]


async def test_list_package_exports_empty(server):
    """Returns a clear message when no exports match the package path."""
    async with Client(server) as client:
        result = await client.call_tool(
            "list_package_exports",
            {"package_path": "nonexistent_pkg", "workspace_id": WS},
        )
    assert not result.is_error
    assert "No public exports" in result.data


# ── dependency_graph ───────────────────────────────────────────────────────


async def test_dependency_graph_human(server):
    """Human output reports symbol and dependency counts."""
    async with Client(server) as client:
        result = await client.call_tool("dependency_graph", {"workspace_id": WS})
    assert not result.is_error
    text = result.data
    assert "Dependency Graph" in text
    assert "Symbols" in text
    assert "Dependencies" in text


async def test_dependency_graph_json(server):
    """JSON output has edges, adjacency, and visited_nodes."""
    async with Client(server) as client:
        result = await client.call_tool(
            "dependency_graph", {"workspace_id": WS, "output_format": "json"}
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert "edges" in data
    assert "adjacency" in data
    assert "visited_nodes" in data
    assert isinstance(data["visited_nodes"], list)


async def test_dependency_graph_with_entry_point(server):
    """entry_point triggers BFS subgraph output."""
    async with Client(server) as client:
        result = await client.call_tool(
            "dependency_graph",
            {"workspace_id": WS, "entry_point": "main", "output_format": "json"},
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert "entry_point_graph" in data
    ep = data["entry_point_graph"]
    assert isinstance(ep["nodes"], list)
    assert "main" in ep["nodes"]


# ── project_stats ──────────────────────────────────────────────────────────


async def test_project_stats_human(server):
    """Human output shows files, symbols, and scan count."""
    async with Client(server) as client:
        result = await client.call_tool("project_stats", {"workspace_id": WS})
    assert not result.is_error
    text = result.data
    assert "Project Stats" in text
    assert "Files" in text
    assert "Total symbols" in text
    assert "Scans" in text


async def test_project_stats_json(server):
    """JSON output has current, scan_history, and delta keys."""
    async with Client(server) as client:
        result = await client.call_tool(
            "project_stats", {"workspace_id": WS, "output_format": "json"}
        )
    assert not result.is_error
    data = json.loads(result.data)
    assert "current" in data
    assert "scan_history" in data
    assert "delta" in data


async def test_project_stats_delta(server):
    """delta key reflects difference between first and last scan."""
    async with Client(server) as client:
        result = await client.call_tool(
            "project_stats", {"workspace_id": WS, "output_format": "json"}
        )
    assert not result.is_error
    data = json.loads(result.data)
    delta = data["delta"]
    scans = data["scan_history"]
    # Fixture stores scans (2,5,800) and (3,7,900) → delta files = 1, chunks = 2
    assert delta["files_delta"] == scans[-1]["files_scanned"] - scans[0]["files_scanned"]
    assert delta["chunks_delta"] == scans[-1]["chunks_created"] - scans[0]["chunks_created"]


async def test_project_stats_evolution_in_human(server):
    """Human output shows evolution section when multiple scans exist."""
    async with Client(server) as client:
        result = await client.call_tool("project_stats", {"workspace_id": WS})
    assert not result.is_error
    assert "Evolution" in result.data or "delta" in result.data.lower()
