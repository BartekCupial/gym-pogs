import networkx as nx
import numpy as np


class SymbolicPOGSAgent:
    def __init__(self):
        """Initialize the symbolic agent for POGS environment."""
        self.known_graph = None
        self.current_node = None
        self.target_node = None
        self.path_to_target = None

    def reset(self, observation):
        """Reset the agent's knowledge when the environment resets."""
        # Initialize a new graph with the number of nodes from the observation
        self.known_graph = nx.Graph()

        # Add all possible nodes
        num_nodes = int(np.sqrt(len(observation["vector"]) - 2))
        for i in range(num_nodes):
            self.known_graph.add_node(i)

        # Extract current and target nodes
        self.current_node = observation["current_node"]
        self.target_node = observation["target_node"]

        # Update the known graph with observable edges
        self._update_known_graph(observation)

        # Compute initial path
        self._compute_path()

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

        # Alternative: use edge_list if available
        if "edge_list" in observation and isinstance(observation["edge_list"], (list, set)):
            for u, v in observation["edge_list"]:
                self.known_graph.add_edge(u, v)

    def _compute_path(self):
        """Compute the path to the target using the known graph."""
        try:
            # Try to find a path to the target using Dijkstra's algorithm
            self.path_to_target = nx.shortest_path(self.known_graph, source=self.current_node, target=self.target_node)
            # Remove the current node from the path
            self.path_to_target = self.path_to_target[1:]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path is found, set path to None
            self.path_to_target = None

    def act(self, observation):
        """Determine the next action based on the current observation."""
        # Update knowledge with new observation
        self._update_known_graph(observation)

        # Recompute path if needed
        if self.path_to_target is None or len(self.path_to_target) == 0:
            self._compute_path()

        # If we have a path, take the next step
        if self.path_to_target and len(self.path_to_target) > 0:
            next_node = self.path_to_target[0]
            self.path_to_target = self.path_to_target[1:]
            return next_node

        # If no path is found, use exploration strategy
        # Find unvisited neighbors of the current node
        neighbors = list(self.known_graph.neighbors(self.current_node))
        if neighbors:
            return np.random.choice(neighbors)

        # If no neighbors are available, return a random action
        # (This is a fallback and should rarely happen)
        return np.random.randint(0, self.known_graph.number_of_nodes())
