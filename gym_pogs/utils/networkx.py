import networkx as nx


def find_furthest_node(G, source, nodes=None):
    # Get shortest path lengths from source to all other nodes
    path_lengths = nx.single_source_shortest_path_length(G, source)

    # If nodes is provided, filter path_lengths to only include those nodes
    if nodes is not None:
        # Create a new dictionary with only the nodes we're interested in
        # Also check that the nodes actually exist in path_lengths
        filtered_lengths = {node: path_lengths[node] for node in nodes if node in path_lengths}

        # If no valid nodes were found, return None
        if not filtered_lengths:
            return None, 0

        # Find the node with maximum distance from the filtered set
        furthest_node = max(filtered_lengths.items(), key=lambda x: x[1])
    else:
        # Consider all nodes
        furthest_node = max(path_lengths.items(), key=lambda x: x[1])

    return furthest_node[0], furthest_node[1]  # Returns (node, distance)
