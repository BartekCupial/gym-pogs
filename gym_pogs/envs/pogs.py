import io
from typing import Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygame
from gymnasium import spaces
from PIL import Image

from gym_pogs.utils.graph_generator import generate_graph

matplotlib.use("Agg")


class POGSEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_nodes: int = 20,
        branching_prob: float = 0.3,
        k_nearest: int = 3,
        include_cycles: bool = False,
        undirected: bool = True,
        max_steps: int = 30,
        render_mode: Optional[str] = None,
        screen_size: int | None = 640,
    ):
        """
        Initialize the POGS environment.

        Args:
            num_nodes: Number of nodes in the graph
            branching_prob: Probability of a node being a branching point
            k_nearest: Observation radius (how many hops away can the agent observe)
            include_cycles: Whether to include cycles in the graph
            undirected: Whether to have directed or undirected graph
            render_mode: The render mode to use. Options are 'human' and 'rgb_array'
        """
        super().__init__()

        assert k_nearest >= 1
        assert num_nodes >= 2

        # Check render mode is valid
        self.render_mode = render_mode

        if self.render_mode is not None:
            assert self.render_mode in self.metadata["render_modes"]

        self.num_nodes = num_nodes
        self.branching_prob = branching_prob
        self.k_nearest = k_nearest
        self.include_cycles = include_cycles
        self.undirected = undirected
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(low=0, high=1, shape=(num_nodes * num_nodes + 2,), dtype=np.float32),
                "current_node": spaces.Discrete(num_nodes),
                "target_node": spaces.Discrete(num_nodes),
                "edge_list": spaces.Sequence(spaces.Tuple([spaces.Discrete(num_nodes), spaces.Discrete(num_nodes)])),
            }
        )

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

    def _generate_graph(self):
        # Generate a new graph
        return generate_graph(
            num_nodes=self.num_nodes,
            branching_prob=self.branching_prob,
            include_cycles=self.include_cycles,
            undirected=self.undirected,
            seed=int(self.np_random.integers(0, 2**31 - 1)),
        )

    def _choose_current_node(self):
        # Choose random start and target nodes
        nodes = list(self.graph.nodes())
        return self.np_random.choice(nodes)

    def _choose_target_node(self):
        # Ensure target is not the same as start and is reachable
        nodes = list(self.graph.nodes())
        possible_targets = [n for n in nodes if n != self.current_node]
        return self.np_random.choice(possible_targets)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.graph: nx.Graph = self._generate_graph()
        self.current_node = self._choose_current_node()
        self.target_node = self._choose_target_node()
        self.pos = nx.spring_layout(self.graph)  # for rendering

        # Reset step counter and visited nodes
        self.steps_taken = 0
        self.visited_nodes = {self.current_node}
        self.observable_edges = set()

        # Update observable edges based on current position
        self._update_observable_edges()
        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.steps_taken += 1

        terminated = False
        truncated = False
        reward = 0

        # Check if action is valid (can only move to connected nodes that are observable)
        if action not in self.graph.neighbors(self.current_node):
            reward = -1.0  # Penalty for invalid move
        else:
            # Move to the selected node
            self.current_node = action
            self.visited_nodes.add(self.current_node)

            # Update observable edges
            self._update_observable_edges()

            # Check if target is reached
            if self.current_node == self.target_node:
                reward = 100.0  # Big reward for reaching target
                terminated = True

        if self.steps_taken >= self.max_steps:
            truncated = True

        # Get observation
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observable_nodes(self):
        """Get the set of nodes that are observable from the current node."""
        observable_nodes = set()

        # Add all nodes within k_nearest hops
        for node in nx.single_source_shortest_path_length(self.graph, self.current_node, cutoff=self.k_nearest):
            observable_nodes.add(node)

        return observable_nodes

    def _update_observable_edges(self):
        """Update the set of edges that are observable from the current position."""
        observable_nodes = self._get_observable_nodes()

        # Add edges between observable nodes
        for u, v in self.graph.edges():
            if u in observable_nodes and v in observable_nodes:
                self.observable_edges.add((u, v))
                self.observable_edges.add((v, u))  # Add both directions

    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            observation: A flattened representation of the observable graph and current/target nodes
        """
        # Create adjacency matrix for observable edges
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for u, v in self.observable_edges:
            adj_matrix[u, v] = 1.0

        # Flatten the adjacency matrix and add current and target nodes
        vector = np.concatenate(
            [adj_matrix.flatten(), np.array([self.current_node, self.target_node], dtype=np.float32)]
        )

        # Create edge list representation
        edge_list = list(self.observable_edges)

        if self.undirected:
            edge_list = set(map(tuple, map(sorted, edge_list)))

        return {
            "vector": vector,
            "current_node": self.current_node,
            "target_node": self.target_node,
            "edge_list": edge_list,
        }

    def _get_info(self):
        return {
            "visited_nodes": len(self.visited_nodes),
            "steps_taken": self.steps_taken,
            "distance_to_target": nx.shortest_path_length(self.graph, self.current_node, self.target_node),
        }

    def get_frame(self):
        G = self.graph

        # Create a list of node colors
        color_map = ["black"] * len(G.nodes())

        color_map[self.target_node] = "blue"
        color_map[self.current_node] = "red"

        plt.figure(figsize=(10, 10))

        # Draw observable nodes with bigger gray circles
        observable_nodes = self._get_observable_nodes()
        nx.draw_networkx_nodes(G, self.pos, nodelist=observable_nodes, node_color="#FFD700", node_size=1500, alpha=0.3)

        # Draw all nodes
        nx.draw_networkx_nodes(G, self.pos, node_color="black", node_size=500)

        # Draw target and current nodes
        nx.draw_networkx_nodes(G, self.pos, nodelist=[self.target_node], node_color="blue", node_size=500)
        nx.draw_networkx_nodes(G, self.pos, nodelist=[self.current_node], node_color="red", node_size=500)

        # Draw edges
        nx.draw_networkx_edges(G, self.pos)

        # Draw labels
        nx.draw_networkx_labels(G, self.pos, font_color="white", font_weight="bold")

        plt.tight_layout()

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()

        # Convert buffer to PIL Image
        buf.seek(0)
        img = Image.open(buf)

        img_array = np.array(img)

        # Convert to RGB array (removing alpha channel if present)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        return img_array

    def render(self):
        img = self.get_frame()

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("pogs")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface((int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset)))
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self):
        if self.window:
            pygame.quit()
