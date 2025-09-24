"""
Knowledge Graph wrapper for Felix Framework web interface.

This module provides a simplified knowledge graph interface that wraps
the existing knowledge_store functionality for web interface compatibility.
"""

from typing import Dict, List, Any, Optional
from .knowledge_store import KnowledgeStore


class KnowledgeGraph:
    """
    Simplified knowledge graph interface for Felix Framework web deployment.

    Wraps the existing KnowledgeStore functionality to provide a graph-like
    interface suitable for web interface integration.
    """

    def __init__(self):
        """Initialize knowledge graph with underlying knowledge store."""
        self.store = KnowledgeStore()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node to the knowledge graph."""
        self.nodes[node_id] = data
        # Also store in underlying knowledge store
        self.store.store_knowledge(node_id, data)

    def add_edge(self, source: str, target: str, relationship: str, weight: float = 1.0) -> None:
        """Add an edge between two nodes."""
        edge = {
            "source": source,
            "target": target,
            "relationship": relationship,
            "weight": weight
        }
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes."""
        neighbors = []
        for edge in self.edges:
            if edge["source"] == node_id:
                neighbors.append(edge["target"])
            elif edge["target"] == node_id:
                neighbors.append(edge["source"])
        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges)
        }

    def clear(self) -> None:
        """Clear all graph data."""
        self.nodes.clear()
        self.edges.clear()
        # Note: We don't clear the underlying store to preserve session data