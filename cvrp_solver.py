import pandas as pd
import numpy as np
import random
import math
import os
from copy import deepcopy
from collections import defaultdict
import time

# --- Configuration & Constants ---
SPEED_KMH = 22.72
R_EARTH_KM = 6371.0

# --- Helper Functions ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = R_EARTH_KM
    return c * r

def load_data(folder_path):
    """
    Load all necessary data from CSV files in the specified folder.
    """
    # Load Clients
    clients_path = os.path.join(folder_path, 'clients.csv')
    clients_df = pd.read_csv(clients_path)
    
    # Load Depots
    depots_path = os.path.join(folder_path, 'depots.csv')
    depots_df = pd.read_csv(depots_path)
    # Assume single depot for now as per instructions (first row)
    depot = depots_df.iloc[0]
    
    # Load Vehicles
    vehicles_path = os.path.join(folder_path, 'vehicles.csv')
    vehicles_df = pd.read_csv(vehicles_path)
    
    # Load Parameters
    # Search for parameters file
    param_files = [f for f in os.listdir(folder_path) if f.startswith('parameters') and f.endswith('.csv')]
    if not param_files:
        raise FileNotFoundError(f"No parameters file found in {folder_path}")
    
    params_path = os.path.join(folder_path, param_files[0])
    params_df = pd.read_csv(params_path)
    params = {}
    for _, row in params_df.iterrows():
        params[row['Parameter']] = float(row['Value'])
        
    return clients_df, depot, vehicles_df, params

# --- Genetic Algorithm Class ---

class GeneticAlgorithmCVRP:
    def __init__(self, clients_df, depot, vehicles_df, params, 
                 population_size=100, generations=500, mutation_rate=0.2, 
                 crossover_rate=0.8, elitism_rate=0.1, tournament_size=5):
        
        self.clients_df = clients_df
        self.depot = depot
        self.vehicles_df = vehicles_df.sort_values(by='Capacity', ascending=False).reset_index(drop=True)
        self.params = params
        
        self.clients = clients_df['ClientID'].tolist()
        self.demands = dict(zip(clients_df['ClientID'], clients_df['Demand']))
        self.coords = dict(zip(clients_df['ClientID'], zip(clients_df['Latitude'], clients_df['Longitude'])))
        
        self.depot_coords = (depot['Latitude'], depot['Longitude'])
        
        # Available vehicles (pool)
        self.fleet_capacities = self.vehicles_df['Capacity'].tolist()
        self.num_vehicles = len(self.fleet_capacities)
        
        # GA Parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Precompute distance matrix (Depot <-> Clients and Clients <-> Clients)
        # 0 index will represent the Depot
        self.all_points = [0] + self.clients
        self.dist_matrix = {}
        
        print("Precomputing distances...")
        # Depot to Clients
        for cid in self.clients:
            c_lat, c_lon = self.coords[cid]
            d = haversine_distance(self.depot_coords[0], self.depot_coords[1], c_lat, c_lon)
            self.dist_matrix[(0, cid)] = d
            self.dist_matrix[(cid, 0)] = d
            
        # Clients to Clients
        for i in range(len(self.clients)):
            for j in range(i + 1, len(self.clients)):
                c1 = self.clients[i]
                c2 = self.clients[j]
                d = haversine_distance(self.coords[c1][0], self.coords[c1][1], 
                                       self.coords[c2][0], self.coords[c2][1])
                self.dist_matrix[(c1, c2)] = d
                self.dist_matrix[(c2, c1)] = d
        print("Distances computed.")

    def get_distance(self, p1, p2):
        if p1 == p2: return 0.0
        return self.dist_matrix.get((p1, p2), 0.0)

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            solution = self._create_random_solution()
            self.population.append(solution)
            
    def _create_random_solution(self):
        """
        Creates a random solution.
        A solution is a list of routes. Each route is a list of client IDs.
        """
        shuffled_clients = self.clients.copy()
        random.shuffle(shuffled_clients)
        
        # Partition into k routes (where k is random but <= num_vehicles)
        num_routes = random.randint(1, self.num_vehicles)
        routes = [[] for _ in range(num_routes)]
        
        for i, client in enumerate(shuffled_clients):
            routes[i % num_routes].append(client)
            
        return routes

    def evaluate_fitness(self, solution):
        """
        Calculate total cost + penalties.
        """
        total_dist_km = 0.0
        total_cost = 0.0
        
        # 1. Assign routes to vehicles
        # Strategy: Sort routes by Demand (Descdending) and match with Vehicles by Capacity (Descending)
        
        route_demands = []
        for route in solution:
            d = sum(self.demands[c] for c in route)
            route_demands.append((d, route))
            
        route_demands.sort(key=lambda x: x[0], reverse=True)
        
        # We have self.fleet_capacities sorted desc in __init__
        
        # Check for vehicle count violation
        if len(solution) > self.num_vehicles:
            # Huge Penalty for using more vehicles than available
            return 1e9 * (len(solution) - self.num_vehicles)
            
        used_vehicles_indices = []
        capacity_penalty = 0.0
        
        # Greedy assignment
        
        for i in range(len(solution)):
            r_demand = route_demands[i][0]
            route = route_demands[i][1]
            # Assigned vehicle capacity
            v_cap = self.fleet_capacities[i]
            
            if r_demand > v_cap:
                # Capacity Violation
                over = r_demand - v_cap
                capacity_penalty += over * 1000000 # Heavy penalty per unit over capacity
            
            # Calculate Route Distance
            r_dist = 0.0
            if not route:
                continue
                
            # Depot -> First
            r_dist += self.get_distance(0, route[0])
            # Inter-client
            for k in range(len(route)-1):
                r_dist += self.get_distance(route[k], route[k+1])
            # Last -> Depot
            r_dist += self.get_distance(route[-1], 0)
            
            total_dist_km += r_dist
            
        # Costs
        # Dist Cost = Dist * C_dist
        # Time Cost = (Dist / Speed) * C_time
        # Fuel Cost = Dist * (Price / Efficiency)
        # Fixed Cost = Num_Routes * C_fixed
        
        c_dist_val = self.params.get('C_dist', 0)
        c_time_val = self.params.get('C_time', 0)
        c_fixed_val = self.params.get('C_fixed', 0)
        fuel_price = self.params.get('fuel_price', 0)
        fuel_eff = self.params.get('fuel_efficiency_typical', 1)
        
        cost_dist = total_dist_km * c_dist_val
        
        total_time_h = total_dist_km / SPEED_KMH
        cost_time = total_time_h * c_time_val
        
        cost_fuel = total_dist_km * (fuel_price / fuel_eff)
        
        cost_fixed = len(solution) * c_fixed_val
        
        total_objective = cost_dist + cost_time + cost_fuel + cost_fixed
        
        return total_objective + capacity_penalty

    # --- Operators (Adapted from notebook) ---
    
    def select_parents(self):
        def tournament():
            participants = random.sample(self.population, self.tournament_size)
            best = min(participants, key=lambda s: self.evaluate_fitness(s))
            return best
        return tournament(), tournament()
        
    def crossover(self, parent1, parent2):
        # Ordered Crossover (OX) on Giant Tour
        
        def to_giant_tour(sol):
            return [c for r in sol for c in r]
            
        gt1 = to_giant_tour(parent1)
        gt2 = to_giant_tour(parent2)
        
        # OX Crossover
        size = len(gt1)
        a, b = sorted(random.sample(range(size), 2))
        
        def ox(p1, p2):
            child = [None]*size
            child[a:b+1] = p1[a:b+1]
            idx = (b+1)%size
            p2_idx = (b+1)%size
            while None in child:
                if p2[p2_idx] not in child:
                    child[idx] = p2[p2_idx]
                    idx = (idx+1)%size
                p2_idx = (p2_idx+1)%size
            return child
            
        c_gt1 = ox(gt1, gt2)
        c_gt2 = ox(gt2, gt1)
        
        # Split back to routes
        child1 = self._split_giant_tour(c_gt1)
        child2 = self._split_giant_tour(c_gt2)
        
        return child1, child2
        
    def _split_giant_tour(self, tour):
        routes = []
        
        fleet = self.fleet_capacities.copy()
        random.shuffle(fleet)
        
        tour_idx = 0
        veh_idx = 0
        
        while tour_idx < len(tour):
            if veh_idx >= len(fleet):
                # Run out of vehicles! Put rest in last route
                if len(routes) > 0:
                    routes[-1].extend(tour[tour_idx:])
                else:
                    routes.append(tour[tour_idx:])
                break
                
            cap = fleet[veh_idx]
            current_route = []
            current_load = 0
            
            while tour_idx < len(tour):
                c = tour[tour_idx]
                d = self.demands[c]
                if current_load + d <= cap:
                    current_route.append(c)
                    current_load += d
                    tour_idx += 1
                else:
                    break
            
            # Use vehicle even if empty? No, ignore empty routes unless forced?
            # If we couldn't fit even one client (too big?), force it to avoid infinite loop
            if not current_route and tour_idx < len(tour):
                 current_route.append(tour[tour_idx])
                 tour_idx += 1
                 
            if current_route:
                routes.append(current_route)
            
            veh_idx += 1
            
        return routes

    def mutate(self, solution):
        if random.random() < self.mutation_rate:
            mut_type = random.choice(['swap', 'move'])
            
            if mut_type == 'swap':
                # Swap two clients from any routes
                r1_idx = random.randint(0, len(solution)-1)
                r2_idx = random.randint(0, len(solution)-1)
                
                if solution[r1_idx] and solution[r2_idx]:
                    i = random.randint(0, len(solution[r1_idx])-1)
                    j = random.randint(0, len(solution[r2_idx])-1)
                    solution[r1_idx][i], solution[r2_idx][j] = solution[r2_idx][j], solution[r1_idx][i]
                    
            elif mut_type == 'move':
                # Take client from r1, put in r2
                r1_idx = random.randint(0, len(solution)-1)
                r2_idx = random.randint(0, len(solution)-1)
                
                if solution[r1_idx] and r1_idx != r2_idx:
                    i = random.randint(0, len(solution[r1_idx])-1)
                    c = solution[r1_idx].pop(i)
                    j = random.randint(0, len(solution[r2_idx])) # Insert pos
                    solution[r2_idx].insert(j, c)
                    
                    if not solution[r1_idx]:
                        solution.pop(r1_idx) # Remove empty route
                        
        return solution

    def evolve(self):
        # 1. Scores
        scored_pop = [(s, self.evaluate_fitness(s)) for s in self.population]
        scored_pop.sort(key=lambda x: x[1])
        
        # Record best
        if scored_pop[0][1] < self.best_fitness:
            self.best_fitness = scored_pop[0][1]
            self.best_solution = deepcopy(scored_pop[0][0])
            
        # Elitism
        elite_count = max(1, int(self.population_size * self.elitism_rate))
        new_pop = [deepcopy(s[0]) for s in scored_pop[:elite_count]]
        
        # Fill
        while len(new_pop) < self.population_size:
            p1, p2 = self.select_parents()
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            new_pop.append(c1)
            if len(new_pop) < self.population_size:
                new_pop.append(self.mutate(c2))
                
        self.population = new_pop
        
    def solve(self):
        self.initialize_population()
        # print(f"Initial Best Fitness: {self.evaluate_fitness(self.population[0]):.2f}")
        
        for g in range(self.generations):
            self.evolve()
            # if g % 50 == 0:
            #     print(f"Generation {g}: Best Fitness = {self.best_fitness:,.2f}")
                
        return self.best_solution, self.best_fitness

# --- Main Execution ---

if __name__ == "__main__":
    # Si está en la carpeta raíz del proyecto (Proyecto-E3), usar la siguiente línea 
    base_path = r"cvrp_content-main"
    cases = ["caso_base", "caso_2", "caso_3"]
    
    for case in cases:
        print(f"\n{'='*20}\nSolving {case}...\n{'='*20}")
        case_path = os.path.join(base_path, case)
        
        if not os.path.exists(case_path):
            print(f"Path not found: {case_path}")
            continue
            
        clients_df, depot, vehicles_df, params = load_data(case_path)
        
        print(f"Loaded {len(clients_df)} clients, {len(vehicles_df)} vehicles from {case}")
        
        ga = GeneticAlgorithmCVRP(
            clients_df, depot, vehicles_df, params,
            population_size=150,
            generations=500, # 500
            mutation_rate=0.3
        )
        
        best_sol, best_fit = ga.solve()
        
        print(f"\nFinal Result for {case}:")
        print(f"Total Cost (Fitness): {best_fit:,.2f}")
        print("Routes:")
        for i, route in enumerate(best_sol):
            print(f"  Route {i+1}: {route} (Len: {len(route)})")
