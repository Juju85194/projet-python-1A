from collections import deque
from heapq import heappop, heappush
from tqdm import tqdm

class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.parent_dictionnary = None

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
        heappush(queue, (0, src, []))

        # Main loop
        while queue:
            dist, node, path = heappop(queue)

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
                    heappush(queue, (new_dist, neighbor, new_path))
        # If no valid path exists, return None
        return None

    def get_path_with_power_BFS(self, src, dest, power):
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
        # We use the BFS algorithm to find the path with the minimum number of edges while maintaining the power condition
        # Initialize visited and queue
        visited = set()
        queue = deque([(src, [])])

        # Main loop
        while queue:
            node, path = queue.popleft()

            # Check if we reached the destination
            if node == dest:
                path.append(node)
                return path

            # Check if the power is sufficient to traverse the edge
            for neighbor, power_min, _ in self.graph[node]:
                if neighbor not in visited and power >= power_min:
                    visited.add(neighbor)
                    new_path = path.copy()
                    new_path.append(node)
                    queue.append((neighbor, new_path))

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

    def kruskal(self):
        """
        Returns the minimum spanning tree (MST) of the graph g using Kruskal's algorithm.
        """
        # Sort the edges by increasing power
        edges = sorted([(power, u, v) for u in self.graph for v, power, _ in self.graph[u]])

        # Initialize an empty MST
        mst = Graph(nodes=self.nodes)
        
        # Initialize the Union-Find data structure
        parent = {node: node for node in self.nodes}
        rank = {node: 0 for node in self.nodes}
        
        # Define the find and union functions for the Union-Find data structure
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                if rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    if rank[root1] == rank[root2]:
                        rank[root1] += 1
        
        # Iterate over the sorted edges and add them to the MST if they don't create a cycle
        for power, u, v in edges:
            if find(u) != find(v):
                mst.add_edge(u, v, power)
                union(u, v)
        
        return mst

    def dfs(self, start, parent=None, depth=0, depth_dict=None):
        """
        Performs a depth-first search from a starting node in a graph, storing the parent and depth of each node.
        Parameters:
        -----------
        graph: Graph object
            The graph to search.
        start: int
            The starting node.
        parent: int, optional (default=None)
            The parent node of the starting node. None if the starting node is the root of the tree.
        depth: int, optional (default=0)
            The depth of the starting node. 0 if the starting node is the root of the tree.
        depth_dict: dict, optional (default=None)
            The dictionary to store the depths of nodes. If None, a new dictionary is created.
        Returns:
        --------
        A dictionary that maps each node to its parent and depth in the tree.
        """
        if depth_dict is None:
            depth_dict = {}
        depth_dict[start] = (parent, depth)
        for neighbor, _, _ in self.graph[start]:
            if neighbor != parent:
                self.dfs(neighbor, start, depth+1, depth_dict)
        return depth_dict

    def get_path_to_lca(self, node1, node2, parent_dict):
        """
        Returns the path from node1 to the least common ancestor (LCA) of node1 and node2, and the path from the LCA to node2.
        """
        path1 = [node1]
        path2 = [node2]
        while node1 != node2:
            if parent_dict[node1][1] < parent_dict[node2][1]:
                node2 = parent_dict[node2][0]
                path2.append(node2)
            else:
                node1 = parent_dict[node1][0]
                path1.append(node1)
        lca = node1
        path = path1 + path2[::-1][1:]  # combine both paths and remove LCA from one of them
        return path
    
    def edge_power(self, node1, node2):
        """
        Returns the power of the edge between node1 and node2 in the given graph.
        """
        for tpl in self.graph[node1]:
            if tpl[0] == node2:
                return tpl[1]
        return None

    def max_puissance_path(self, p) :

        max_power = float('-inf')
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            
            max_power = max(max_power, self.edge_power(u, v))
                        
        return max_power

    def min_power_mst(self, src, dest):
        """
        Should return path, min_power.

        CAUTION: This method must only be used if the graph is a MST
        """
        if self.parent_dictionnary == None :
            self.parent_dictionnary = self.dfs(1)

        p = self.get_path_to_lca(src, dest, self.parent_dictionnary)
        max_power = self.max_puissance_path(p)
        
        return p, max_power



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
            values = list(map(float, line.split()))
            node1, node2, power_min = values[:3]
            if len(values) == 4:
                dist = values[3]
            else:
                dist = 1
            graph.add_edge(int(node1), int(node2), power_min, dist)

    return graph
