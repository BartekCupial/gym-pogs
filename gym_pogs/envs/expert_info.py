import gymnasium as gym
import networkx as nx
import numpy as np


class ExpertInfo(gym.Wrapper):
    def __init__(self, env, expert_penalty: float = -1):
        super().__init__(env)
        assert expert_penalty <= 0, "expert_penalty must be <= 0"

        self.radius = self.env.unwrapped.k_nearest
        self.expert_penalty = expert_penalty

        self.known_graph: nx.Graph = None

    def _update_known_graph(self, observation):
        """Update the agent's knowledge of the graph based on observation."""
        initial_node_count = self.known_graph.number_of_nodes()
        self.added_nodes = False

        # Extract adjacency matrix from the observation vector
        num_nodes = int(np.sqrt(len(observation["vector"]) - 2))
        adj_matrix = observation["vector"][:-2].reshape(num_nodes, num_nodes)

        if len(self.history) > 1 and self.unwrapped.current_node == self.history[-2]:
            self.backtrack_count += 1

        if self.previous_node is not None or self.previous_node != self.unwrapped.current_node:
            self.history.append(self.unwrapped.current_node)

        # Update current node
        self.previous_node = self.unwrapped.current_node  # Store previous node before updating

        # Check for backtracking
        # If we moved to a previously visited node, that's backtracking
        # if self.unwrapped.current_node in self.visited_nodes and not self.backtracking:
        #     self.backtrack_count += 1
        #     self.backtracking = True
        # elif self.unwrapped.current_node not in self.visited_nodes:
        #     self.backtracking = False

        self.visited_nodes.add(self.unwrapped.current_node)

        # Add edges from the adjacency matrix to the known graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    if i not in self.known_graph:
                        self.known_graph.add_node(i)
                    if j not in self.known_graph:
                        self.known_graph.add_node(j)

                    self.known_graph.add_edge(i, j)

        # Check if new nodes were added
        final_node_count = self.known_graph.number_of_nodes()
        if final_node_count > initial_node_count:
            self.added_nodes = True

    def _update_effectively_explored_nodes(self):
        assert self.radius

        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.unwrapped.current_node)
        effectively_explored_nodes = {node for node, length in path_lengths.items() if length <= self.radius - 1}
        self.explored_nodes.update(effectively_explored_nodes)

    def _is_node_within_radius(self, node):
        """Check if a node is within the exploration radius."""
        return nx.shortest_path_length(self.known_graph, self.unwrapped.current_node, node) <= self.radius - 1

    def _compute_target_path(self):
        """Compute the path to the target using the known graph."""
        try:
            # Try to find a path to the target using Dijkstra's algorithm
            path = nx.shortest_path(
                self.known_graph, source=self.unwrapped.current_node, target=self.unwrapped.target_node
            )
            # Remove the current node from the path
            path = path[1:]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path is found, set path to None
            path = None

        return path

    def _compute_explore_paths(self):
        self.explore_count += 1

        # Find nodes that are just outside our exploration radius
        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.unwrapped.current_node)

        # Find nodes that are outside our explored radius but still known
        frontier_nodes = {node for node in self.known_graph.nodes() if node not in self.explored_nodes}

        if not frontier_nodes:
            # this should only happen if we are done
            return []

        # Find the minimum distance to a frontier node
        min_distance = min(path_lengths.get(node, float("inf")) for node in frontier_nodes)

        # Find all frontier nodes at this minimum distance
        closest_frontier_nodes = [
            node for node in frontier_nodes if path_lengths.get(node, float("inf")) == min_distance
        ]

        # Compute shortest paths to all closest frontier nodes
        all_paths = []
        for target in closest_frontier_nodes:
            path = nx.shortest_path(
                self.known_graph,
                source=self.unwrapped.current_node,
                target=target,
            )
            # Remove the current node from the path
            path = path[1:]
            if len(path) > 0:
                all_paths.append(path)

        assert len(all_paths) > 0, "No valid paths found"
        return all_paths

    def _expert_move(self):
        expert_action = set()

        # if target, expert action is goto target node
        path_to_target = self._compute_target_path()
        if path_to_target:
            expert_action.add(path_to_target[0])
            return expert_action

        # if no target, expert action is goto closest boundary node (or more then one)
        paths_to_explore = self._compute_explore_paths()
        for path in paths_to_explore:
            expert_action.add(path[0])

        return list(expert_action)

    def _detect_dead_end_discovery(self):
        """
        Detect if we just discovered a dead end

        This happens when:
        1. We just now observed a node which before was a frontier node, but now it isn't and it deosn't have neighbors
        """
        # Get nodes that are within our observation radius
        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.unwrapped.current_node)
        node_degree = nx.degree(self.known_graph)
        effectively_explored_nodes = {node for node, length in path_lengths.items() if length <= self.radius - 1}
        dead_ends = {node for node in effectively_explored_nodes if node_degree[node] == 1}
        new_dead_ends = dead_ends - self.dead_ends_discovered

        # Update our set of discovered dead ends
        self.dead_ends_discovered.update(new_dead_ends)

        # Return True if we found any new dead ends
        return len(new_dead_ends) > 0, list(new_dead_ends)

    def _detect_target_discovery(self):
        """
        Detect if we just discovered the target

        This happens when:
        1. The target node becomes visible in the known graph
        2. There was a possibility that our agent was exploring other path of the graph
        """
        # If the target is not in the known graph, we haven't discovered it
        if self.unwrapped.target_node not in self.known_graph:
            return False

        # Report target discovery no more then once
        if self.target_discovered:
            return False
        self.target_discovered = True

        # If we see the target for the first time it should be one of the frontier nodes
        path_lengths = nx.single_source_shortest_path_length(self.known_graph, self.unwrapped.current_node)
        edge_nodes = {
            node
            for node in self.known_graph.nodes()
            if node not in self.explored_nodes and path_lengths[node] == self.radius
        }
        assert self.unwrapped.target_node in edge_nodes

        # If we could detec target accidentally count it as darget discovery
        return len(edge_nodes) > 1

    def _update_expert_matches(self, action):
        info = self.last_info["episode_extra_stats"]

        if action in info["expert_action"]:
            self.expert_matches += 1
        # else:
        #     print("error: action not in expert action")
        self.total_actions += 1

        if info["dead_end_discovery"] or info["target_discovery"]:
            if action in info["expert_action"]:
                self.expert_hard_matches += 1
            self.total_hard_actions += 1

    def _expert_accuracy(self):
        return self.expert_matches / self.total_actions if self.total_actions > 0 else 1

    def _expert_hard_accuracy(self):
        return self.expert_hard_matches / self.total_hard_actions if self.total_hard_actions > 0 else 1

    def _expert_info(self, info):
        episode_extra_stats = info.get("episode_extra_stats", {})

        episode_extra_stats["expert_action"] = self._expert_move()
        episode_extra_stats["target_discovery"] = self._detect_target_discovery()
        episode_extra_stats["dead_end_discovery"], info["new_dead_ends"] = self._detect_dead_end_discovery()
        episode_extra_stats["expert_accuracy"] = self._expert_accuracy()
        episode_extra_stats["expert_hard_accuracy"] = self._expert_hard_accuracy()
        episode_extra_stats["total_hard_actions"] = self.total_hard_actions
        episode_extra_stats["added_nodes"] = self.added_nodes
        episode_extra_stats["backtrack_count"] = self.backtrack_count

        info["episode_extra_stats"] = episode_extra_stats

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.known_graph = nx.Graph()
        self.visited_nodes = set()
        self.explored_nodes = set()

        self.explore_count = 0
        self.backtrack_count = 0
        self.backtracking = False
        self.previous_node = None

        self.target_discovered = False
        self.dead_ends_discovered = set()
        self.history = []

        # Update knowledge with new observation
        self._update_known_graph(obs)
        self._update_effectively_explored_nodes()

        # Reset expert action tracking
        self.expert_matches = 0
        self.expert_hard_matches = 0
        self.total_actions = 0
        self.total_hard_actions = 0

        # update info
        self._expert_info(info)
        self.last_info = info

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        reward += 0 if action in self.last_info["episode_extra_stats"]["expert_action"] else self.expert_penalty

        # Update knowledge with new observation
        self._update_known_graph(obs)
        self._update_effectively_explored_nodes()

        # only run this in step
        self._update_expert_matches(action)

        # update info
        self._expert_info(info)
        self.last_info = info

        return obs, reward, term, trun, info
