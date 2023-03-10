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
    graph = graph_from_file(data_path + filename_network)
    # Load the data from the route file
    with open(data_path + filename_route, "r") as f:
        total_nb_trip = int(f.readline())
        data = f.readlines()
    # Create a list of trips
    all_trip = []
    for line in data:
        src, dest, _ = map(int, line.split())
        all_trip.append([src,dest])
    # Choose some random trips
    nb_trip = 5
    chosen_trip = []
    for _ in range(nb_trip):
        i = np.random.randint(0,total_nb_trip)
        chosen_trip.append(all_trip[i])
    # Measure the time taken for each chosen trip
    times = []
    for trip in chosen_trip:
        start_time = time.perf_counter()
        graph.min_power(trip[0], trip[1])
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    # Calculate the average time taken
    avg_time = sum(times) / len(times)
    # Calculate the estimated total time
    est_total_time = avg_time * total_nb_trip
    print(f"Estimated time for {filename_route}: {est_total_time:.2f} seconds")

# estimate_time('routes.1.in', 'network.1.in')
# estimate_time('routes.2.in', 'network.2.in')
# estimate_time('routes.3.in', 'network.3.in')
# estimate_time('routes.4.in', 'network.4.in')
# estimate_time('routes.5.in', 'network.5.in')
# estimate_time('routes.6.in', 'network.6.in')
# estimate_time('routes.7.in', 'network.7.in')
# estimate_time('routes.8.in', 'network.8.in')
# estimate_time('routes.9.in', 'network.9.in')
# estimate_time('routes.10.in', 'network.10.in')


######################### Question 15 ############################

def estimate_time_mst(filename_route, filename_network):
    # Create the graph
    graph = graph_from_file(data_path + filename_network)
    # Measure the time taken
    start_time = time.perf_counter()
    # Load the data from the route file
    with open(data_path + filename_route, "r") as f:
        total_nb_trip = int(f.readline())
        data = f.readlines()
    # Create a list of trips
    all_trip = []
    for line in data:
        src, dest, _ = map(int, line.split())
        all_trip.append([src, dest])
    # Calculate the minimum power for each trip
    min_powers = []
    for trip in all_trip:
        min_powers.append(graph.min_power_mst(trip[0], trip[1])[1])
    # Write the minimum powers to a file
    filename_out = filename_route.replace(".in", ".out")
    with open('output/'+filename_out, "w") as f:
        for min_power in min_powers:
            f.write(f"{min_power}\n")
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print(f"Processed {filename_route} in {time_taken:.2f} seconds")

estimate_time_mst('routes.1.in', 'network.1.in')
# estimate_time_mst('routes.2.in', 'network.2.in')
# estimate_time_mst('routes.3.in', 'network.3.in')
# estimate_time_mst('routes.4.in', 'network.4.in')
# estimate_time_mst('routes.5.in', 'network.5.in')
# estimate_time_mst('routes.6.in', 'network.6.in')
# estimate_time_mst('routes.7.in', 'network.7.in')
# estimate_time_mst('routes.8.in', 'network.8.in')
# estimate_time_mst('routes.9.in', 'network.9.in')
# estimate_time_mst('routes.10.in', 'network.10.in')
