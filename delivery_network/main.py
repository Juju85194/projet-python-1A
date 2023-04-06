from graph import Graph, graph_from_file
import numpy as np
from tqdm import tqdm
import numpy as np
import copy


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
    for trip in tqdm(chosen_trip):
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

    return trucks


def min_powers_from_file(path):
    """Returns a list of minimum powers
    Args:
        path (string): path to the file containing the minimum powers
    """
    minimum_powers = []
    with open(path, "r") as f:
        data = f.readlines()
    for line in data:
        power = float(line)
        minimum_powers.append(power)
    return minimum_powers

# A brute force solution to the problem

def sort_routes_and_min_powers(routes, minimum_powers):
    """Returns routes sorted by profit and the power required for each route
    Args:
        routes (list): list of trips
        minimum_powers (list): list of minimum powers
    """
    routes, minimum_powers = zip(*sorted(zip(routes, minimum_powers), key=lambda x: x[0][2], reverse=True))
    routes = list(routes)
    minimum_powers = list(minimum_powers)
    return routes, minimum_powers


def max_profit(routes,trucks,minimum_powers):
    """Returns the maximum profit achievable using the trucks, without budget constraint.
    Args:
        graph (Graph): graph of the network
        routes (list): list of trips (src, dest, profit) SORTED by descending profit
        trucks (int list): list of trucks (power, cost)
        minimum_powers (list) : a list containing the minimum powers each trip
    Returns:
        int: maximum_profit
    """
    alocated_trips = []
    maximum_profit = 0

    # Sort trucks by power
    trucks = sorted(trucks)

    # Main iteration
    while trucks:
        truck = trucks.pop(0)
        for i in range(len(routes)):
            if truck[0] >= minimum_powers[i] and i not in alocated_trips:
                alocated_trips.append(i)
                maximum_profit += routes[i][2]
                break

    return maximum_profit


def solve(routes, trucks, budget, minimum_powers):
    """Brute force the exact solution of the problem
    Args:
        routes (list): list of SORTED trips by profit
        trucks (list): list of trucks (power, cost)
        budget (int): budget
        minimum_powers (list): list powers required for each trip
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
        if len(new_combination) <= len(routes):
            combinations.extend(new_combination)

    # At least one truck is chosen
    combinations.pop(0)

    # Go through each combination of trucks to find the optimal solution
    for combination in combinations:
        # Compute the cost of the combination and its profit
        sum_cost = sum([cost for _, cost in combination])
        profit = max_profit(routes, combination, minimum_powers)

        # Check if the cost doesn't exceed the budget and if the profit is larger
        if sum_cost <= budget and profit > best_profit:
            best_profit = profit
            best_solution = combination

    return best_solution


def duplicate_trucks(trucks,budget):
    """Duplicate trucks to account for unlimited stock
    Args:
        trucks (list): list of trucks
        budget (int): budget
    Returns:
        list: list of trucks duplicated
    """
    total_cost = 0
    duplicated_trucks = []

    for truck in trucks:
        while total_cost < budget:
            total_cost += truck[1]
            duplicated_trucks.append(truck)
        total_cost = 0

    return duplicated_trucks

# The previous solution is not able to find the answer in a decent span of time.


# Greedy approach

def trip_to_truck(trip_index, trucks , minimum_powers):
    """Returns the best truck suited for a given trip_index
    Args:
        trip_index (int) : index of the trip
        trucks (list) : (indice_truck, power, cost). List of simplified trucks.
        min_powers (list): list of minimum powers required for each trip
    """
    power_needed = minimum_powers[trip_index]

    # If the most powerful truck isn't able to handle the trip, return (-1, 0, 0)
    if power_needed > trucks[-1][1] :
        return (-1, 0, 0)

    # Find the best truck using binary search
    debut, fin = 0, len(trucks) - 1
    while debut <= fin:
        milieu = (debut + fin) // 2
        if power_needed > trucks[milieu][1]:
            debut = milieu + 1
        else:
            fin = milieu - 1

    if debut > 0 and power_needed == trucks[debut-1][1] :
        return trucks[debut-1]
    else :
        return trucks[debut]


def simplified_trucks(trucks) :
    """
    Simplifies the list of trucks : trucks that are less powerful but more expensive than other trucks are deleted.
    Note: trucks are already sorted by increasing power in truck files.
    """
    # Create a copy of trucks and add indexes
    temp_trucks = copy.deepcopy(trucks)
    for i, tup in enumerate(temp_trucks):
        temp_trucks[i] = [i] + tup
    
    current_cost = temp_trucks[-1][2]  # The cost of the most powerful truck

    # Iterate over the trucks in reverse order (except the most powerful truck)
    for i in range(len(temp_trucks)-2, -1, -1):
        if temp_trucks[i][2] > current_cost:
            del temp_trucks[i]
        else:
            current_cost = temp_trucks[i][2]

    return temp_trucks


def ratio(routes, trucks, minimum_powers):
    """Returns a list of (trip_index, ratio, truck used to achieve such a ratio) sorted by ascending profit per unit of cost.

    Args:
        routes (list): list of routes
        trucks (list): list of simplified trucks
        min_powers (list): list of minimum power required for each trip
    """
    ratios = []
    for index_route, route in enumerate(routes):
        profit = route[2]
        truck = trip_to_truck(index_route, trucks, minimum_powers)
        truck_cost = truck[2]
        r = profit/truck_cost
        ratios.append((route, r, truck))

    ratios = sorted(ratios, key=lambda x: x[1])
    return ratios


def solve_greedy(routes, trucks, budget, minimum_powers):
    """Returns a solution to the problem using a greedy approach
    Args:
        routes (list): list of routes
        trucks (list): list of trucks
        budget (int): budget
        min_powers (list): list of minimum power required for each trip
    Output:
        solution(list) : list of (truck, route)
    """
    solution = []
    ratios = ratio(routes, simplified_trucks(trucks), minimum_powers)
    updated_budget = budget

    while updated_budget > 0 and ratios:

        r = ratios.pop()

        if r[2][2] <= updated_budget:
            updated_budget -= r[2][2]
            solution.append(((r[2][1], r[2][2]), r[0]))
        # If we don't have enough money, we just go on to the next trip

    return solution


b = 25*1e9
g = graph_from_file(data_path + 'network.10.in')
g = g.kruskal()
routes = routes_from_file('routes.10.in')
trucks = trucks_from_file('trucks.2.in')
minimum_powers = min_powers_from_file('output/routes.10.out')


# Brute force
# trucks = duplicate_trucks(trucks, b)
# solution = solve(routes, trucks, b, minimum_powers)
# print(solution)

# Greedy approximation
solution = solve_greedy(routes, trucks, b, minimum_powers)
print(solution)
