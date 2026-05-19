"""Dependency analysis for code workspaces using graph algorithms."""

import sqlite3
from collections import deque, defaultdict
from typing import Optional, Any

from mcp_code_rag.storage import Storage


class DependencyAnalyzer:
    """Analyzes dependencies between code symbols in a workspace."""

    def __init__(self, storage: Storage):
        """
        Initialize DependencyAnalyzer with a Storage backend.

        Args:
            storage: Storage instance for accessing dependency_index.
        """
        self.storage = storage

    def build_dependency_graph(
        self,
        workspace_id: str,
        max_depth: Optional[int] = None,
        entry_point: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Build a dependency graph from the dependency_index.

        If entry_point is specified, performs BFS from that node up to max_depth.
        Otherwise, returns the full graph.

        Returns:
            dict with keys:
            - edges: list of (source, target) tuples
            - adjacency: dict mapping source -> list of targets
            - reverse_adjacency: dict mapping target -> list of sources
            - entry_point_graph: BFS subgraph if entry_point specified
            - visited_nodes: set of nodes in the graph
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch all dependencies for workspace
        cursor.execute(
            "SELECT source_symbol, target_symbol FROM dependency_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        edges = [(row["source_symbol"], row["target_symbol"]) for row in rows]

        # Build adjacency list
        adjacency = defaultdict(list)
        reverse_adjacency = defaultdict(list)
        visited_nodes = set()

        for source, target in edges:
            adjacency[source].append(target)
            reverse_adjacency[target].append(source)
            visited_nodes.add(source)
            visited_nodes.add(target)

        result = {
            "edges": edges,
            "adjacency": dict(adjacency),
            "reverse_adjacency": dict(reverse_adjacency),
            "visited_nodes": visited_nodes,
        }

        # If entry_point specified, perform BFS
        if entry_point:
            entry_graph = self._bfs_subgraph(
                entry_point, adjacency, visited_nodes, max_depth
            )
            result["entry_point_graph"] = entry_graph

        return result

    def _bfs_subgraph(
        self,
        entry_point: str,
        adjacency: dict,
        all_nodes: set,
        max_depth: Optional[int],
    ) -> dict[str, Any]:
        """
        Perform BFS from entry_point up to max_depth.

        Returns:
            dict with keys: nodes, edges, depths (node -> distance from entry_point)
        """
        if entry_point not in all_nodes:
            return {"nodes": set(), "edges": [], "depths": {}}

        visited = {entry_point}
        queue = deque([(entry_point, 0)])
        edges = []
        depths = {entry_point: 0}

        while queue:
            node, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for target in adjacency.get(node, []):
                if target not in visited:
                    visited.add(target)
                    depths[target] = depth + 1
                    queue.append((target, depth + 1))
                    edges.append((node, target))

        return {"nodes": visited, "edges": edges, "depths": depths}

    def detect_cycles(self, workspace_id: str) -> list[list[str]]:
        """
        Detect cycles in the dependency graph using DFS with coloring.

        Colors:
        - white (0): unvisited
        - gray (1): in progress
        - black (2): finished

        Returns:
            List of cycles, each cycle is a list of symbols.
        """
        graph = self.build_dependency_graph(workspace_id)
        adjacency = graph["adjacency"]
        all_nodes = graph["visited_nodes"]

        color = {node: 0 for node in all_nodes}
        parent = {node: None for node in all_nodes}
        cycles = []

        def dfs(node: str, path: list[str]) -> None:
            color[node] = 1  # gray
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if color[neighbor] == 1:  # back edge (cycle detected)
                    # Extract cycle from path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif color[neighbor] == 0:  # white
                    dfs(neighbor, path)

            path.pop()
            color[node] = 2  # black

        for node in all_nodes:
            if color[node] == 0:
                dfs(node, [])

        return cycles

    def get_callers(self, symbol_name: str, workspace_id: str) -> list[dict[str, Any]]:
        """
        Find all symbols that import/call the given symbol.

        Returns:
            List of dicts with keys: source_symbol, file_path
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find dependencies where symbol_name is the target
        cursor.execute(
            """
            SELECT DISTINCT di.source_symbol, si.file_path
            FROM dependency_index di
            LEFT JOIN symbol_index si ON si.workspace_id = di.workspace_id
                AND si.symbol_name = di.source_symbol
            WHERE di.target_symbol = ? AND di.workspace_id = ?
            """,
            (symbol_name, workspace_id),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_targets(self, symbol_name: str, workspace_id: str) -> list[dict[str, Any]]:
        """
        Find all symbols that the given symbol imports/uses.

        Returns:
            List of dicts with keys: target_symbol, file_path
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find dependencies where symbol_name is the source
        cursor.execute(
            """
            SELECT DISTINCT di.target_symbol, si.file_path
            FROM dependency_index di
            LEFT JOIN symbol_index si ON si.workspace_id = di.workspace_id
                AND si.symbol_name = di.target_symbol
            WHERE di.source_symbol = ? AND di.workspace_id = ?
            """,
            (symbol_name, workspace_id),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def most_imported(self, workspace_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Find the most imported symbols (targets with most callers).

        Returns:
            List of dicts with keys: symbol, import_count
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT target_symbol as symbol, COUNT(*) as import_count
            FROM dependency_index
            WHERE workspace_id = ?
            GROUP BY target_symbol
            ORDER BY import_count DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [{"symbol": row[0], "import_count": row[1]} for row in rows]

    def most_dependent(self, workspace_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Find the most dependent symbols (sources with most targets).

        Returns:
            List of dicts with keys: symbol, dependency_count
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT source_symbol as symbol, COUNT(*) as dependency_count
            FROM dependency_index
            WHERE workspace_id = ?
            GROUP BY source_symbol
            ORDER BY dependency_count DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [{"symbol": row[0], "dependency_count": row[1]} for row in rows]

    def format_edges(self, graph: dict) -> list[tuple[str, str]]:
        """Return edges as list of (source, target) tuples."""
        return graph["edges"]

    def format_adjacency(self, graph: dict) -> dict:
        """Return adjacency list representation."""
        return graph["adjacency"]

    def format_visual(self, graph: dict) -> str:
        """Return ASCII visual representation of the graph."""
        lines = []
        adjacency = graph["adjacency"]

        lines.append("Dependency Graph Visualization:")
        lines.append("=" * 50)

        for source, targets in sorted(adjacency.items()):
            lines.append(f"{source}")
            for target in sorted(targets):
                lines.append(f"  └─> {target}")

        if not adjacency:
            lines.append("(empty graph)")

        return "\n".join(lines)

    def format_summary(self, graph: dict, workspace_id: str) -> dict:
        """Return summary statistics about the graph."""
        edges = graph["edges"]
        adjacency = graph["adjacency"]
        reverse_adj = graph["reverse_adjacency"]
        visited_nodes = graph["visited_nodes"]

        total_symbols = len(visited_nodes)
        total_dependencies = len(edges)

        # Calculate in/out degrees
        in_degrees = {node: len(reverse_adj.get(node, [])) for node in visited_nodes}
        out_degrees = {node: len(adjacency.get(node, [])) for node in visited_nodes}

        most_imported = max((node for node in visited_nodes), key=lambda n: in_degrees[n]) if visited_nodes else None
        most_dependent = max((node for node in visited_nodes), key=lambda n: out_degrees[n]) if visited_nodes else None

        return {
            "total_symbols": total_symbols,
            "total_dependencies": total_dependencies,
            "avg_dependencies_per_symbol": (
                total_dependencies / total_symbols if total_symbols > 0 else 0
            ),
            "most_imported": most_imported,
            "most_imported_count": in_degrees.get(most_imported, 0) if most_imported else 0,
            "most_dependent": most_dependent,
            "most_dependent_count": out_degrees.get(most_dependent, 0) if most_dependent else 0,
        }
