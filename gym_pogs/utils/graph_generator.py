import random

import networkx as nx
import numpy as np


def generate_graph(num_nodes=20, branching_prob=0.3, include_cycles=False, undirected=True, seed=None):
    """
    Generate a random graph for the POGS environment.

    Args:
        num_nodes: Number of nodes in the graph
        branching_prob: Probability of a node being a branching point
        include_cycles: Whether to include cycles in the graph
        k_nearest: Number of nearest neighbors to connect in the initial graph

    Returns:
        G: A networkx graph
    """
    if include_cycles:
        # TODO:
        assert False
    else:
        # Create a tree (no cycles)
        G = nx.random_labeled_tree(num_nodes, seed=seed)
        assert undirected, "For now only supports undirected"

    # Adjust the graph based on branching probability
    # TODO:

    # Relabel nodes to be integers from 0 to num_nodes-1
    G = nx.convert_node_labels_to_integers(G)

    return G
