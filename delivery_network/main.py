from graph import Graph, graph_from_file
import numpy as np
from tqdm import tqdm
import numpy as np


data_path = "input/"
# file_name = "network.01.in"

# g = graph_from_file(data_path + file_name)
# print(g)

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
        src, dest, _ = map(float, line.split())
        all_trip.append([int(src),int(dest)])
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
    # Create the MST
    mst = graph.kruskal()
    # Measure the time taken
    start_time = time.perf_counter()
    # Load the data from the route file
    with open(data_path + filename_route, "r") as f:
        total_nb_trip = int(f.readline())
        data = f.readlines()
    # Create a list of trips
    all_trip = []
    for line in data:
        src, dest, _ = map(float, line.split())
        all_trip.append([int(src), int(dest)])
    # Calculate the minimum power for each trip
    min_powers = []
    for trip in tqdm(all_trip):
        min_powers.append(mst.min_power_mst(trip[0], trip[1])[1])
    # Write the minimum powers to a file
    filename_out = filename_route.replace(".in", ".out")
    with open('output/'+filename_out, "w") as f:
        for min_power in min_powers:
            f.write(f"{min_power}\n")
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print(f"Processed {filename_route} in {time_taken:.2f} seconds")

# estimate_time_mst('routes.1.in', 'network.1.in')
# estimate_time_mst('routes.2.in', 'network.2.in')
# estimate_time_mst('routes.3.in', 'network.3.in')
# estimate_time_mst('routes.4.in', 'network.4.in')
# estimate_time_mst('routes.5.in', 'network.5.in')
# estimate_time_mst('routes.6.in', 'network.6.in')
# estimate_time_mst('routes.7.in', 'network.7.in')
# estimate_time_mst('routes.8.in', 'network.8.in')
# estimate_time_mst('routes.9.in', 'network.9.in')
# estimate_time_mst('routes.10.in', 'network.10.in')


######################### Question 18 ############################

# A brute force solution to the problem

def max_profit(graph,routes,trucks):
    """Returns the maximum profit achievable using the trucks, without budget constraint.

    Args:
        graph (Graph): graph of the network
        routes (list): list of trips (src, dest, profit)
        trucks (int list): list of trucks (power, cost)

    Returns:
        int: maximum_profit
    """
    alocated_trips = []
    maximum_profit = 0

    # Sort trips by descending profit
    routes = sorted(routes, key=lambda x: x[2], reverse=True)

    # Sort trucks by power
    trucks = sorted(trucks)

    # Compute required power for each trip
    minimum_powers = [graph.min_power_mst(src,dest) for src, dest, _ in routes]
    
    # Main iteration
    while trucks:
        truck = trucks.pop(0)
        for i in range(len(routes)):
            if truck[0] >= minimum_powers[i] and i not in alocated_trips:
                alocated_trips.append(i)
                maximum_profit += routes[i][2]
                break
        
    return maximum_profit
    

def solve(graph, routes, trucks, budget):
    """Brute force the exact solution of the problem

    Args:
        trucks (list): list of trucks (power, cost)
        budget (int): budget

    Returns:
        list: collection of trucks
    """
    combinations = [[]]
    best_solution = []

    # Initialize best profit
    best_profit = 0

    # Get every combination of trucks
    for truck in trucks:
        new_combination = [combination + [truck] for combination in combinations]
        combination.extend(new_combination)
    
    # At least one truck is chosen
    combinations.pop(0)

    # Go through each combination of trucks to find the optimal solution
    for combination in combinations:
        # Compute the cost of the combination and its profit
        sum_cost = sum([cost for _, cost in combination])
        profit = max_profit(graph, routes, combination)

        # Check if the cost doesn't exceed the budget and if the profit is larger
        if sum_cost <= budget and profit > best_profit:
            best_profit = profit
            best_solution = combination

    return best_solution

def routes_from_file(filename_route):
    """Make a list of routes from a file

    Args:
        filename_route (string): name of the file

    Returns:
        list: a list of routes (src, dest, profit)
    """ 
    routes = []

    # Load the data from the route file
    with open(data_path + filename_route, "r") as f:
        f.readline()
        data = f.readlines()
    
    for line in data:
        src, dest, profit = map(float, line.split())
        routes.append([int(src), int(dest), profit])
    
    return routes

def trucks_from_file(filename_truck):
    """Make a list of trucks from a file

    Args:
        filename_truck (string): name of the file

    Returns:
        list: a list of trucks (power, cost)
    """ 
    trucks = []

    # Load the data from the truck file
    with open(data_path + filename_truck, "r") as f:
        f.readline()
        data = f.readlines()
    
    for line in data:
        power, cost = map(float, line.split())
        trucks.append([power, cost])
    
    return routes

g = graph_from_file(data_path + 'network.1.in')
g = g.kruskal()
routes = routes_from_file('routes.1.in')
trucks = trucks_from_file('trucks.0.in')
b = 25*10e9

solve(g, routes, trucks, b)