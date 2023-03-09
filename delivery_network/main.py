from graph import Graph, graph_from_file
import numpy as np


data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)

######################### Question 10 ############################

import time

def estimate_time(filename_route, filename_network):
    # Create the graph
    graph = graph_from_file(filename_network)
    # Load the data from the route file
    with open(filename_route, "r") as f:
        total_nb_path = int(f.readline())
        data = f.readlines()
    # Create a list of path
    all_path = []
    for line in data:
        src, dest, _ = map(int, line.split())
        all_path.append([src,dest])
    # Choose some random path
    nb_path = 1
    chosen_path = []
    for _ in range(nb_path):
        i = np.random.randint(0,total_nb_path)
        chosen_path.append(all_path[i])
    # Measure the time taken for each chosen path
    times = []
    for path in chosen_path:
        start_time = time.perf_counter()
        graph.min_power(path[0], path[1])
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    # Calculate the average time taken
    avg_time = sum(times) / len(times)
    # Calculate the estimated total time
    est_total_time = avg_time * total_nb_path
    print(f"Estimated time for {filename_route}: {est_total_time:.2f} seconds")

estimate_time('input/routes.1.in', 'input/network.1.in')
# estimate_time('input/routes.2.in', 'input/network.2.in')
# estimate_time('input/routes.3.in', 'input/network.3.in')
# estimate_time('input/routes.4.in', 'input/network.4.in')
# estimate_time('input/routes.5.in', 'input/network.5.in')
# estimate_time('input/routes.6.in', 'input/network.6.in')
# estimate_time('input/routes.7.in', 'input/network.7.in')
# estimate_time('input/routes.8.in', 'input/network.8.in')
# estimate_time('input/routes.9.in', 'input/network.9.in')
# estimate_time('input/routes.10.in', 'input/network.10.in')

