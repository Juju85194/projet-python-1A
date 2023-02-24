import heapq
import graphviz

class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1

    def get_path_with_power(self, src, dest, power):
        """
        Determines if a truck with power p can cover the path t (and returns a valid path if it's possible). 

        Parameters: 
        -----------
        src: NodeType
            Starting node of the path
        dest: NodeType
            End node of the path
        power: numeric (int or float)
            Power of the truck 

        Returns:
        -----------
        path: list or None
            A list of nodes in the path (including src and dest) if a valid path exists, None otherwise.
        """
        # We use the Dijkstra algorithm to find the path with the shortest distance while maintaining the power condition
        # Initialize distances to all nodes to infinity
        distances = {node: float('inf') for node in self.graph}
        distances[src] = 0

        # Initialize priority queue and add the source node
        queue = []
        heapq.heappush(queue, (0, src, []))

        # Main loop
        while queue:
            dist, node, path = heapq.heappop(queue)

            # Check if we reached the destination
            if node == dest:
                path.append(node)
                return path

            # Check if we have already explored this node with a shorter distance
            if dist > distances[node]:
                continue

            # Check if the power is sufficient to traverse the edge
            for neighbor, power_min, edge_dist in self.graph[node]:
                new_dist = dist + edge_dist
                if new_dist < distances[neighbor] and power >= power_min:
                    distances[neighbor] = new_dist
                    new_path = path + [node]
                    heapq.heappush(queue, (new_dist, neighbor, new_path))

        # If no valid path exists, return None
        return None
        

    def connected_components(self):
        """
        Returns a list of connected components in the graph.
        """
        visited = set()
        components = []
        for node in self.graph:
            if node not in visited:
                component = []
                self._dfs(node, visited, component)
                components.append(component)
        return components

    def _dfs(self, node, visited, component):
        """
        A recursive helper function for the connected_components method.
        """
        visited.add(node)
        component.append(node)
        for neighbor, _, _ in self.graph[node]:
            if neighbor not in visited:
                self._dfs(neighbor, visited, component)

    def connected_components_set(self):
        components = self.connected_components()
        return set(map(frozenset, components))


    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        """
        # Compute the maximum possible power needed to traverse the graph
        max_power = sum(power for _, power, _ in sum(self.graph.values(), []))
        # Binary search to find the minimum required power
        left, right = 1, max_power
        while left < right:
            mid = (left + right) // 2
            if self.get_path_with_power(src, dest, mid):
                right = mid
            else:
                left = mid + 1
        # Return the path and the minimum power
        return self.get_path_with_power(src, dest, left), left


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    nodes = []
    graph = Graph(nodes)

    with open(filename) as f:
        n, m = map(int, f.readline().split())
        nodes = list(range(1, n + 1))
        graph = Graph(nodes)

        for line in f:
            values = list(map(int, line.split()))
            node1, node2, power_min = values[:3]
            if len(values) == 4:
                dist = values[3]
            else:
                dist = 1
            graph.add_edge(node1, node2, power_min, dist)

    return graph

def visualize_graph(self, src, dest):
    dot = graphviz.Graph(comment='Truck problem')

    # Add nodes and edges
    for i in range(self.nb_nodes):
        node = self.nodes[i]
        dot.node(str(node))
        for neighbor, power, distance in self.graph[node]:
            dot.edge(str(node), str(neighbor), label=f"{power}/{neighbor}")
    
    # Highlight source and destination nodes
    dot.node(str(src), style='filled', fillcolor='green')
    dot.node(str(dest), style='filled', fillcolor='red')

    # Highlight path
    path, _ = self.min_power(src, dest)
    for i in range(len(path) - 1):
        dot.edge(str(path[i]), str(path[i+1]), color='blue')

    return dot