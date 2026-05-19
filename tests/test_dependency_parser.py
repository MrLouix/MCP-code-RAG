"""Tests for dependency analysis using DependencyAnalyzer."""

import pytest

from mcp_code_rag.storage import Storage
from mcp_code_rag.dependency import DependencyAnalyzer


@pytest.fixture
def storage(tmp_path):
    """Create a Storage instance with temp directory."""
    return Storage(str(tmp_path))


@pytest.fixture
def analyzer(storage):
    """Create a DependencyAnalyzer instance."""
    return DependencyAnalyzer(storage)


class TestBuildDependencyGraph:
    """Tests for build_dependency_graph method."""

    def test_graph_three_files_correct_edges(self, storage, analyzer):
        """Verify graph with 3 files has correct number of edges."""
        workspace_id = "test_ws"

        # Setup: 3 files with dependencies
        # file_a.py imports file_b.py
        storage.store_dependency("file_a.py:main", "file_b.py:func", workspace_id)
        # file_b.py imports file_c.py
        storage.store_dependency("file_b.py:func", "file_c.py:helper", workspace_id)
        # file_a.py also imports file_c.py
        storage.store_dependency("file_a.py:main", "file_c.py:helper", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)

        # Should have exactly 3 edges
        assert len(graph["edges"]) == 3
        assert len(graph["visited_nodes"]) == 3

        # Check edges exist
        edges_set = set(graph["edges"])
        assert ("file_a.py:main", "file_b.py:func") in edges_set
        assert ("file_b.py:func", "file_c.py:helper") in edges_set
        assert ("file_a.py:main", "file_c.py:helper") in edges_set

    def test_adjacency_list_structure(self, storage, analyzer):
        """Verify adjacency list is correctly built."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("a", "c", workspace_id)
        storage.store_dependency("b", "c", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)
        adjacency = graph["adjacency"]

        assert "a" in adjacency
        assert set(adjacency["a"]) == {"b", "c"}
        assert "b" in adjacency
        assert adjacency["b"] == ["c"]

    def test_reverse_adjacency(self, storage, analyzer):
        """Verify reverse adjacency (reverse dependencies) is built."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("c", "b", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)
        reverse_adj = graph["reverse_adjacency"]

        # "b" has incoming edges from "a" and "c"
        assert "b" in reverse_adj
        assert set(reverse_adj["b"]) == {"a", "c"}

    def test_bfs_from_entry_point(self, storage, analyzer):
        """Verify BFS traversal from entry point."""
        workspace_id = "test_ws"

        # Create chain: a -> b -> c -> d
        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "c", workspace_id)
        storage.store_dependency("c", "d", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id, max_depth=2, entry_point="a")

        assert "entry_point_graph" in graph
        entry_graph = graph["entry_point_graph"]
        assert "a" in entry_graph["nodes"]
        assert "b" in entry_graph["nodes"]
        assert "c" in entry_graph["nodes"]
        # d should not be in the graph due to max_depth=2
        assert "d" not in entry_graph["nodes"]


class TestCycleDetection:
    """Tests for detect_cycles method."""

    def test_detect_cycle_simple(self, storage, analyzer):
        """Verify simple cycle detection (a -> b -> a)."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "a", workspace_id)

        cycles = analyzer.detect_cycles(workspace_id)

        assert len(cycles) > 0
        # Cycle should contain both a and b
        all_in_cycles = []
        for cycle in cycles:
            all_in_cycles.extend(cycle)
        assert "a" in all_in_cycles
        assert "b" in all_in_cycles

    def test_detect_cycle_three_node(self, storage, analyzer):
        """Verify cycle detection in three-node cycle (a -> b -> c -> a)."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "c", workspace_id)
        storage.store_dependency("c", "a", workspace_id)

        cycles = analyzer.detect_cycles(workspace_id)

        assert len(cycles) > 0

    def test_acyclic_graph_returns_empty(self, storage, analyzer):
        """Verify acyclic graph returns empty cycle list."""
        workspace_id = "test_ws"

        # Create acyclic graph: a -> b -> c (no cycle back)
        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "c", workspace_id)

        cycles = analyzer.detect_cycles(workspace_id)

        assert cycles == []

    def test_empty_graph_no_cycles(self, storage, analyzer):
        """Verify empty graph has no cycles."""
        workspace_id = "test_ws"

        cycles = analyzer.detect_cycles(workspace_id)

        assert cycles == []


class TestGetCallers:
    """Tests for get_callers method."""

    def test_get_callers_finds_importing_files(self, storage, analyzer):
        """Verify get_callers returns files that import a symbol."""
        workspace_id = "test_ws"

        # Store symbol locations
        from mcp_code_rag.chunker import CodeChunk

        chunk_a = CodeChunk(
            type="function",
            name="func_a",
            package="a",
            language="python",
            file_path="a.py",
            start_line=1,
            end_line=5,
            selection_start=1,
            selection_end=1,
            signature="def func_a():",
            code="def func_a():\n    pass",
        )

        chunk_b = CodeChunk(
            type="function",
            name="func_b",
            package="b",
            language="python",
            file_path="b.py",
            start_line=1,
            end_line=5,
            selection_start=1,
            selection_end=1,
            signature="def func_b():",
            code="def func_b():\n    pass",
        )

        storage.store_chunk(chunk_a, [0.1] * 768, workspace_id)
        storage.store_chunk(chunk_b, [0.1] * 768, workspace_id)

        # func_a imports func_b
        storage.store_dependency("func_a", "func_b", workspace_id)
        # func_c also imports func_b
        storage.store_dependency("func_c", "func_b", workspace_id)

        callers = analyzer.get_callers("func_b", workspace_id)

        assert len(callers) == 2
        caller_names = {c["source_symbol"] for c in callers}
        assert "func_a" in caller_names
        assert "func_c" in caller_names

    def test_get_callers_empty_for_unused(self, storage, analyzer):
        """Verify get_callers returns empty list for unused symbols."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)

        callers = analyzer.get_callers("unused_symbol", workspace_id)

        assert callers == []


class TestGetTargets:
    """Tests for get_targets method."""

    def test_get_targets_finds_dependencies(self, storage, analyzer):
        """Verify get_targets returns symbols that a symbol depends on."""
        workspace_id = "test_ws"

        from mcp_code_rag.chunker import CodeChunk

        chunk_b = CodeChunk(
            type="function",
            name="func_b",
            package="b",
            language="python",
            file_path="b.py",
            start_line=1,
            end_line=5,
            selection_start=1,
            selection_end=1,
            signature="def func_b():",
            code="def func_b():\n    pass",
        )

        chunk_c = CodeChunk(
            type="function",
            name="func_c",
            package="c",
            language="python",
            file_path="c.py",
            start_line=1,
            end_line=5,
            selection_start=1,
            selection_end=1,
            signature="def func_c():",
            code="def func_c():\n    pass",
        )

        storage.store_chunk(chunk_b, [0.1] * 768, workspace_id)
        storage.store_chunk(chunk_c, [0.1] * 768, workspace_id)

        # func_a depends on func_b and func_c
        storage.store_dependency("func_a", "func_b", workspace_id)
        storage.store_dependency("func_a", "func_c", workspace_id)

        targets = analyzer.get_targets("func_a", workspace_id)

        assert len(targets) == 2
        target_names = {t["target_symbol"] for t in targets}
        assert "func_b" in target_names
        assert "func_c" in target_names

    def test_get_targets_empty_for_leaf_node(self, storage, analyzer):
        """Verify get_targets returns empty for symbols with no dependencies."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)

        targets = analyzer.get_targets("b", workspace_id)

        assert targets == []


class TestMostImported:
    """Tests for most_imported method."""

    def test_most_imported_ranking(self, storage, analyzer):
        """Verify most_imported returns symbols ranked by import count."""
        workspace_id = "test_ws"

        # b is imported 3 times, c is imported 2 times, d is imported once
        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("x", "b", workspace_id)
        storage.store_dependency("y", "b", workspace_id)
        storage.store_dependency("p", "c", workspace_id)
        storage.store_dependency("q", "c", workspace_id)
        storage.store_dependency("r", "d", workspace_id)

        results = analyzer.most_imported(workspace_id, limit=10)

        assert len(results) >= 1
        assert results[0]["symbol"] == "b"
        assert results[0]["import_count"] == 3


class TestMostDependent:
    """Tests for most_dependent method."""

    def test_most_dependent_ranking(self, storage, analyzer):
        """Verify most_dependent returns symbols ranked by dependency count."""
        workspace_id = "test_ws"

        # a depends on 3 symbols, b depends on 2, c depends on 1
        storage.store_dependency("a", "x", workspace_id)
        storage.store_dependency("a", "y", workspace_id)
        storage.store_dependency("a", "z", workspace_id)
        storage.store_dependency("b", "p", workspace_id)
        storage.store_dependency("b", "q", workspace_id)
        storage.store_dependency("c", "r", workspace_id)

        results = analyzer.most_dependent(workspace_id, limit=10)

        assert len(results) >= 1
        assert results[0]["symbol"] == "a"
        assert results[0]["dependency_count"] == 3


class TestFormatting:
    """Tests for formatting methods."""

    def test_format_edges(self, storage, analyzer):
        """Verify format_edges returns edge list."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "c", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)
        edges = analyzer.format_edges(graph)

        assert len(edges) == 2
        assert ("a", "b") in edges
        assert ("b", "c") in edges

    def test_format_adjacency(self, storage, analyzer):
        """Verify format_adjacency returns dict representation."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("a", "c", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)
        adjacency = analyzer.format_adjacency(graph)

        assert isinstance(adjacency, dict)
        assert "a" in adjacency
        assert set(adjacency["a"]) == {"b", "c"}

    def test_format_visual(self, storage, analyzer):
        """Verify format_visual returns ASCII text representation."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "c", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)
        visual = analyzer.format_visual(graph)

        assert isinstance(visual, str)
        assert "a" in visual
        assert "b" in visual
        assert "c" in visual

    def test_format_summary(self, storage, analyzer):
        """Verify format_summary returns statistics."""
        workspace_id = "test_ws"

        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("b", "c", workspace_id)

        graph = analyzer.build_dependency_graph(workspace_id)
        summary = analyzer.format_summary(graph, workspace_id)

        assert isinstance(summary, dict)
        assert "total_symbols" in summary
        assert "total_dependencies" in summary
        assert summary["total_symbols"] == 3
        assert summary["total_dependencies"] == 2
