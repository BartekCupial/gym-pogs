from typing import List

import networkx as nx
import numpy as np

from gym_pogs.utils.networkx import find_furthest_node


class MemorySymbolicPOGSAgent:
    def __init__(self):
        """Initialize the symbolic agent for POGS environment."""
        self.known_graph: nx.Graph = None
        self.current_node: int = None
        self.target_node: int = None
        self.path_to_target: List[int] = None

    def reset(self, observation):
        """Reset the agent's knowledge when the environment resets."""
        # Initialize a new graph with the number of nodes from the observation
        self.known_graph = nx.Graph()
        self.visited_nodes = set()

        # Add all possible nodes
        num_nodes = int(np.sqrt(len(observation["vector"]) - 2))
        for i in range(num_nodes):
            self.known_graph.add_node(i)

        # Extract current and target nodes
        self.current_node = observation["current_node"]
        self.target_node = observation["target_node"]

        # Update the known graph with observable edges
        self._update_known_graph(observation)

        furthest_node, distance = find_furthest_node(self.known_graph, self.current_node)
        self.radius = distance

        # self.visited_nodes.add(self.current_node)

        self._update_effectively_explored_nodes()

        self.path_to_target = None
        self.path_to_explore = None

        self.num_explore_paths = 0

    def _update_known_graph(self, observation):
        """Update the agent's knowledge of the graph based on observation."""
        # Extract adjacency matrix from the observation vector
        num_nodes = int(np.sqrt(len(observation["vector"]) - 2))
        adj_matrix = observation["vector"][:-2].reshape(num_nodes, num_nodes)

        # Update current node
        self.current_node = observation["current_node"]

        # Add edges from the adjacency matrix to the known graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    self.known_graph.add_edge(i, j)

    def _update_effectively_explored_nodes(self):
        assert self.radius

        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.current_node)
        effectively_explored_nodes = {node for node, length in path_lengths.items() if length <= self.radius - 1}
        self.visited_nodes.update(effectively_explored_nodes)

    def _compute_target_path(self):
        """Compute the path to the target using the known graph."""
        try:
            # Try to find a path to the target using Dijkstra's algorithm
            path = nx.shortest_path(self.known_graph, source=self.current_node, target=self.target_node)
            # Remove the current node from the path
            path = path[1:]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path is found, set path to None
            path = None

        return path

    def _compute_explore_path(self):
        self.num_explore_paths += 1

        unvisited_nodes = set(self.known_graph.nodes()) - self.visited_nodes
        assert len(unvisited_nodes) > 0, "graph should be solvable"
        furthest_unvisited_node, _ = find_furthest_node(self.known_graph, self.current_node, nodes=unvisited_nodes)

        path = nx.shortest_path(
            self.known_graph,
            source=self.current_node,
            target=furthest_unvisited_node,
        )
        # Remove the current node from the path
        path = path[1:]
        assert len(path) > 0
        return path

    def _follow_path(self, path_attribute_name):
        # If we have a path, take the next step
        path = getattr(self, path_attribute_name)
        if path and len(path) > 0:
            next_node = path[0]
            setattr(self, path_attribute_name, path[1:])
            return next_node
        else:
            return None

    def act(self, observation):
        """Determine the next action based on the current observation."""
        # Update knowledge with new observation
        self._update_known_graph(observation)
        self._update_effectively_explored_nodes()

        # Recompute path if needed
        if self.path_to_target is None:
            assert (
                self.path_to_target is None or len(self.path_to_target) > 0
            ), "if path to target == 0, we should be on target"
            self.path_to_target = self._compute_target_path()
            if self.path_to_target:
                self.path_to_explore = None

        next_node = self._follow_path("path_to_target")
        if next_node is not None:
            return next_node

        if self.path_to_explore is None or len(self.path_to_explore) == 0:
            self.path_to_explore = self._compute_explore_path()

        return self._follow_path("path_to_explore")
