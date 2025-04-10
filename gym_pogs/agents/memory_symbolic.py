from typing import List

import networkx as nx
import numpy as np


class MemorySymbolicPOGSAgent:
    def __init__(self, k_nearest):
        self.radius = k_nearest

        self.known_graph: nx.Graph = None
        self.current_node: int = None
        self.target_node: int = None
        self.path_to_target: List[int] = None

    def reset(self):
        # Initialize a new graph with the number of nodes from the observation
        self.known_graph = nx.Graph()
        self.visited_nodes = set()
        self.explored_nodes = set()

        self.path_to_target = None
        self.path_to_explore = None

        self.explore_count = 0
        self.backtrack_count = 0
        self.backtracking = False
        self.previous_node = None

    def _update_known_graph(self, observation):
        """Update the agent's knowledge of the graph based on observation."""
        # Extract adjacency matrix from the observation vector
        num_nodes = int(np.sqrt(len(observation["vector"]) - 2))
        adj_matrix = observation["vector"][:-2].reshape(num_nodes, num_nodes)

        # Update current node
        self.previous_node = self.current_node  # Store previous node before updating
        self.current_node = observation["current_node"]
        self.target_node = observation["target_node"]

        # Check for backtracking
        # If we moved to a previously visited node, that's backtracking
        if self.current_node in self.visited_nodes and not self.backtracking:
            self.backtrack_count += 1
            self.backtracking = True
        elif self.current_node not in self.visited_nodes:
            self.backtracking = False

        self.visited_nodes.add(self.current_node)

        # Add edges from the adjacency matrix to the known graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    if i not in self.known_graph:
                        self.known_graph.add_node(i)
                    if j not in self.known_graph:
                        self.known_graph.add_node(j)

                    self.known_graph.add_edge(i, j)

    def _update_effectively_explored_nodes(self):
        assert self.radius

        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.current_node)
        effectively_explored_nodes = {node for node, length in path_lengths.items() if length <= self.radius - 1}
        self.explored_nodes.update(effectively_explored_nodes)

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
        self.explore_count += 1

        # Find nodes that are just outside our exploration radius
        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.current_node)

        # Find nodes that are outside our explored radius but still known
        frontier_nodes = {node for node in self.known_graph.nodes() if node not in self.explored_nodes}

        if not frontier_nodes:
            assert False, "graph should be solvable"

        # Find the closest node among the frontier nodes
        closest_frontier_node = min(frontier_nodes, key=lambda node: path_lengths.get(node, float("inf")))

        path = nx.shortest_path(
            self.known_graph,
            source=self.current_node,
            target=closest_frontier_node,
        )

        # Remove the current node from the path
        path = path[1:]
        assert len(path) > 0
        return path

    def _is_node_within_radius(self, node):
        """Check if a node is within the exploration radius."""
        return nx.shortest_path_length(self.known_graph, self.current_node, node) <= self.radius - 1

    def act(self, observation):
        """Determine the next action based on the current observation."""
        # Update knowledge with new observation
        self._update_known_graph(observation)
        self._update_effectively_explored_nodes()

        # Try to find a path to target
        if self.path_to_target is None:
            self.path_to_target = self._compute_target_path()
            if self.path_to_target:
                self.path_to_explore = None

        # If we have a path to target, follow it
        if self.path_to_target:
            next_node = self.path_to_target[0]
            self.path_to_target = self.path_to_target[1:]
            return next_node

        # If we don't have an exploration path or the current exploration target is within radius, recompute
        if self.path_to_explore is None or (
            self.path_to_explore and self._is_node_within_radius(self.path_to_explore[-1])
        ):
            self.path_to_explore = self._compute_explore_path()

        # Follow the exploration path
        if self.path_to_explore:
            next_node = self.path_to_explore[0]
            self.path_to_explore = self.path_to_explore[1:]
            return next_node

        assert False, "No valid path found"
