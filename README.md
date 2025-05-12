# gym-pogs

Partially Observable Graph Search

## POGS Environment

The goal of the environment is to find and reach the target node in the partially observable graph, given the information about currently visible part of the graph, current node and target node. The agent has to plan and systematically explore graph to find the target node.

Observation:
- adjacency matrix
- current node, target node
- list of edges

Action space:
- label of the node we want to travel to

### Basics

Make and initialize an environment

```
import gymnasium as gym
import gym_pogs

env = gym.make("HardPOGS-v0", num_nodes=20, k_nearest=2, max_steps=30, min_backtracks=2)
obs, info = env.reset(seed=20)
env.render()
```

### Customize environments

You can specify:
1. number of nodes
2. max_steps
3. k-nearest neighbor observability radius
4. min_backtracks for `HardPOGS-v0`

### Render
image

![image](trajectory.gif)

### Image description

Very easy to convert observation to text. For example to give it to LLMs:

```python
def get_text_observation(obs):
    num_nodes = int(np.sqrt(len(obs["vector"]) - 2))
    adj_matrix = obs["vector"][:-2].reshape(num_nodes, num_nodes)
    description = ""
    for i, row in enumerate(adj_matrix):
        if any(row):
            neigbhors = row.nonzero()[0].tolist()
            description += f"node: {i}, neighbors: {neigbhors}\n"

    obsv = f"{description}\ncurrent node: {obs['current_node']}, target node: {obs['target_node']}"

    return obsv

print(get_text_observation(obs))
```

```
node: 2, neighbors: [4, 16]
node: 3, neighbors: [5, 7, 16]
node: 4, neighbors: [2]
node: 5, neighbors: [3]
node: 7, neighbors: [3]
node: 16, neighbors: [2, 3]

current node: 16, target node: 11
```

### Installation 
The command to install the repository via pip is:
```
pip install git+https://github.com/BartekCupial/gym-pogs
```