import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import simps
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
from functools import partial
import time
import itertools
import argparse

def complex_A1(n_A, n_B, A_AA, A_AB, A_BA, A_BB, n_max):
    """
    Compute A1 with complex cage-type dependent interactions.
    """
    return (n_A * (n_A * A_AA + n_B * A_AB) + n_B * (n_A * A_BA + n_B * A_BB)) / (n_max ** 2)

class Node:
    def __init__(self, ntype, position, topology, cage_type):
        self.ntype = ntype
        self.position = np.array(position)
        self.free_directions = np.arange(topology["n_directions"][ntype])
        self.topology = topology
        self.n_number = 0
        self.n_list = []
        self.nid = None
        self.state = 0
        self.wet_neighbours = {'A': 0, 'B': 0}
        self.surface = False
        self.cage_type = cage_type
        self.hash = str(np.round(self.position, 3))
        self.n_max = topology["n_max"][ntype]

    def update_id(self, nid):
        self.nid = nid

    def add_neigh(self, node, direction):
        if node not in self.n_list:
            self.n_list.append(node)
            self.n_number += 1
            self.free_directions = np.setdiff1d(self.free_directions, direction)
        
        if self.n_number > self.n_max:
            raise ValueError("Maximum number of neighbors exceeded")

    def merge(self, node):
        for neigh in node.n_list:
            if neigh not in self.n_list:
                self.n_list.append(neigh)
                neigh.n_list = [n if n != node else self for n in neigh.n_list]
        
        self.free_directions = np.union1d(self.free_directions, node.free_directions)
        for key in self.wet_neighbours:
            self.wet_neighbours[key] = max(self.wet_neighbours[key], node.wet_neighbours[key])
        self.surface = self.surface or node.surface
        self.n_number = len(self.n_list)

    def surface_node(self):
        self.state = 1
        self.surface = True
        for node in self.n_list:
            node.wet_neighbours['A' if self.cage_type == 0 else 'B'] += 1

    def update_state(self, change):
        if change and not self.surface:
            dif = 1 - self.state * 2
            self.state = 1 - self.state
            for n in self.n_list:
                n.wet_neighbours['A' if self.cage_type == 0 else 'B'] += dif

class Network:
    def __init__(self, topology, type_b_ratio=0.2, A1_values=None, params=None):
        self.lastid = 1
        self.nodes = {}
        self.topology = topology
        self.states = None
        self.rates = None
        self.table_rates = None
        self.dt = None
        self.type_b_ratio = type_b_ratio
        self.average_filling = []
        self.pressures = None
        self.A1_values = A1_values or {'AA': 1.0, 'BB': 0.1, 'AB': 0.5, 'BA': 0.5}
        self.surface_nodes = []
        self.params = params or {}

    def add_node(self, node):
        node.update_id(self.lastid)
        self.nodes[self.lastid] = node
        self.lastid += 1

    def get_surface(self):
        return [node for node in self.nodes.values() if node.n_number < node.n_max]

    def grow(self, n):
        for _ in range(n):
            newnetwork = {"nodes": [], "fathers": []}
            for node in self.nodes.values():
                while len(node.free_directions) > 0:
                    ntype = np.random.choice(self.topology["type_accept"][node.ntype])
                    ndirection = node.free_directions[-1]
                    node.free_directions = node.free_directions[:-1]
                    nposition = node.position + np.array(self.topology["directions"][ndirection]) * self.topology["rotation"][ntype]
                    cage_type = 1 if np.random.rand() < self.type_b_ratio else 0
                    new = Node(ntype, nposition, self.topology, cage_type)
                    ndir = self.topology["reverse"][ndirection]
                    new.add_neigh(node, ndir)
                    newnetwork["nodes"].append([new, node, ndirection])
            
            for nodepair in newnetwork["nodes"]:
                node, old, direction = nodepair
                position = node.position
                ntype = node.ntype
                there = False
                for onode in self.nodes.values():
                    oposition = onode.position
                    dist = np.sum((position - oposition)**2)
                    if dist < 0.2:
                        there = True
                        onode.merge(node)
                        break
                if not there:
                    self.add_node(node)
                    old.add_neigh(node, direction)

    def default_function(self, pressures, params):
        nmax = self.topology["n_max"][0]
        wbarrier_a = params["wbarrier_a"]
        dbarrier_a = params["dbarrier_a"]
        wbarrier_b = params["wbarrier_b"]
        dbarrier_b = params["dbarrier_b"]
        t0 = params["t0"]
        vw_a = params["vw_a"]
        vd_a = params["vd_a"]
        vw_b = params["vw_b"]
        vd_b = params["vd_b"]
        
        prates = np.zeros((len(pressures), 2, nmax + 1, nmax + 1, 2))
        
        for p_idx, p in enumerate(pressures):
            for cage_type in [0, 1]:
                for n in range(nmax + 1):
                    for n_a in range(n + 1):
                        n_b = n - n_a
                        
                        A1 = complex_A1(n_a, n_b, 
                                        self.A1_values['AA'], self.A1_values['AB'],
                                        self.A1_values['BA'], self.A1_values['BB'],
                                        nmax)
                        
                        barrier_w = wbarrier_a if cage_type == 0 else wbarrier_b
                        barrier_d = dbarrier_a if cage_type == 0 else dbarrier_b
                        vw = vw_a if cage_type == 0 else vw_b
                        vd = vd_a if cage_type == 0 else vd_b
                        
                        omegaw = max(barrier_w - p * vw + (nmax/2 - n) * A1, 0)
                        omegad = max(barrier_d + p * vd + (n - nmax/2) * A1, 0)
                        
                        w = 1 / t0 * np.exp(-omegaw)
                        d = 1 / t0 * np.exp(-omegad)
                        prates[p_idx, cage_type, n, n_a] = [w, d]
        
        return prates

    def tabulate_rates(self, func, pressures, params):
        self.table_rates = func(pressures, params)

    def initialize_simulation(self, pressures, dt, func, params):
        self.tabulate_rates(func, pressures, params)
        self.dt = dt
        self.pressures = pressures
        for n in self.nodes.values():
            n.state = 0
            n.wet_neighbours = {'A': 0, 'B': 0}
        
        self.surface_nodes = self.get_surface()
        for n in self.surface_nodes:
            n.surface_node()
        
        self.states = np.array([n.state for n in self.nodes.values()])
        self.update_rates(0)

    def update_states(self, change):
        for ni, n in enumerate(self.nodes.values()):
            if n.surface or any(neighbor.state == 1 for neighbor in n.n_list):
                n.update_state(change[ni])
        self.states = np.array([n.state for n in self.nodes.values()])

    def update_rates(self, t):
        fill = []
        empty = []
        nmax = self.topology["n_max"][0]
        
        for n in self.nodes.values():
            if n.surface or any(neighbor.state == 1 for neighbor in n.n_list):
                cage_type = n.cage_type
                n_a = n.wet_neighbours['A']
                n_b = n.wet_neighbours['B']
                n_total = min(n_a + n_b, nmax)
                
                fill.append(self.table_rates[t][cage_type][n_total][n_a][0])
                empty.append(self.table_rates[t][cage_type][n_total][n_a][1])
            else:
                fill.append(0)
                empty.append(0)
        
        self.rates = np.array([fill, empty])

    def get_random(self):
        return np.random.rand(len(self.table_rates), len(self.nodes))

    def simulate_collective(self):
        dt = self.dt
        self.average_filling = []
        randomnumbers = self.get_random()
        for t in range(len(self.table_rates)):
            states = self.states
            rates = self.rates
            random = randomnumbers[t]
            acceptance = 1 - np.exp(-states*rates[1]*dt - (1-states)*rates[0]*dt)
            change = random < acceptance
            self.update_states(change)
            self.update_rates(t)
            self.average_filling.append(np.mean(states))

def calculate_hysteresis(network):
    pressures = network.pressures
    average_filling = network.average_filling
    half_len = len(pressures) // 2

    intrusion_pressures = pressures[:half_len]
    extrusion_pressures = pressures[half_len:]
    intrusion_filling = average_filling[:half_len]
    extrusion_filling = average_filling[half_len:]

    f_extrusion = interp1d(extrusion_pressures, extrusion_filling, bounds_error=False, fill_value="extrapolate")
    extrusion_filling_aligned = f_extrusion(intrusion_pressures)

    area = simps(np.abs(intrusion_filling - extrusion_filling_aligned), intrusion_pressures)
    return area, intrusion_pressures, extrusion_pressures, intrusion_filling, extrusion_filling

def find_full_saturation_step(network):
    for step, filling in enumerate(network.average_filling):
        if filling == 1.0:
            return step, network.pressures[step]
    return None, None

def store_output_data(network, filename='cage_filling_data.csv'):
    data = {
        'Time_Step': np.arange(len(network.average_filling)),
        'Pressure': network.pressures,
        'Average_Filling': network.average_filling,
        'Nodes_With_8_Neighbors': count_nodes_with_8_neighbors(network),
        'Surface_Nodes': len(network.get_surface()),
        'Total_Nodes': len(network.nodes)
    }

    df = pd.DataFrame(data)
    df['Filling_Rate'] = df['Average_Filling'].diff() / df['Time_Step'].diff()
    df['Pressure_Derivative'] = df['Pressure'].diff() / df['Time_Step'].diff()

    window_size = 10
    df['MA_Filling'] = df['Average_Filling'].rolling(window=window_size).mean()
    df['MA_Filling_Rate'] = df['Filling_Rate'].rolling(window=window_size).mean()

    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

    return df

def run_simulation(type_b_ratio, topology, params, pressures, dt, a_values, growth_steps, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    seed_node = Node(0, [0, 0, 0], topology, cage_type=0)
    network = Network(topology, type_b_ratio=type_b_ratio, A1_values=a_values, params=params)
    network.add_node(seed_node)
    network.grow(growth_steps)

    network.initialize_simulation(pressures, dt, network.default_function, params)
    network.simulate_collective()

    return network

def count_nodes_with_8_neighbors(network):
    return sum(1 for node in network.nodes.values() if node.n_number == 8)

def analyze_surface_distribution(network):
    surface_nodes = network.get_surface()
    total_surface = len(surface_nodes)
    a_type_surface = sum(1 for node in surface_nodes if node.cage_type == 0)
    b_type_surface = total_surface - a_type_surface
    
    return {
        'total_surface': total_surface,
        'a_type_surface': a_type_surface,
        'b_type_surface': b_type_surface,
        'a_type_percentage': a_type_surface/total_surface*100 if total_surface > 0 else 0,
        'b_type_percentage': b_type_surface/total_surface*100 if total_surface > 0 else 0
    }

def process_results(network, type_b_ratio, growth_steps, sim_label, output_dir,
                    case_id=0, parameter='baseline', perturbation=0.0,
                    baseline_value=np.nan, perturbed_value=np.nan,
                    random_seed=None):
    hysteresis_value, *_ = calculate_hysteresis(network)
    saturation_step, saturation_pressure = find_full_saturation_step(network)
    surface_distribution = analyze_surface_distribution(network)
    
    nodes_with_8_neighbors = count_nodes_with_8_neighbors(network)
    surface_nodes = len(network.get_surface())
    total_nodes = len(network.nodes)
    
    # Store output data
    output_file = os.path.join(output_dir, f'cage_filling_data_{sim_label}.csv')
    output_data = store_output_data(network, filename=output_file)

    results = {
        'case_id': case_id,
        'parameter': parameter,
        'perturbation_percent': 100.0 * perturbation,
        'baseline_value': baseline_value,
        'perturbed_value': perturbed_value,
        'random_seed': random_seed,
        'simulation_label': sim_label,
        'growth_steps': growth_steps,
        'type_b_ratio': type_b_ratio,
        'hysteresis': hysteresis_value,
        'saturation_step': saturation_step,
        'saturation_pressure': saturation_pressure,
        'maximum_filling': np.max(network.average_filling),
        'final_filling': network.average_filling[-1],
        'total_nodes': total_nodes,
        'nodes_with_8_neighbors': nodes_with_8_neighbors,
        'nodes_with_8_neighbors_percentage': nodes_with_8_neighbors/total_nodes*100 if total_nodes > 0 else 0,
        'surface_nodes': surface_nodes,
        'surface_nodes_percentage': surface_nodes/total_nodes*100 if total_nodes > 0 else 0,
        'surface_distribution': surface_distribution
    }

    return results

def run_parallel_simulation(sim_config, topology, params, pressures, dt, a_values,
                            base_output_dir, case_id=0, parameter='baseline',
                            perturbation=0.0, baseline_value=np.nan,
                            perturbed_value=np.nan, base_seed=42):
    growth_steps, sim_label, type_b_ratio = sim_config
    start_time = time.time()
    
    # Create output directory for this growth step
    output_dir = os.path.join(base_output_dir, f'growth_{growth_steps}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting simulation {sim_label} with growth={growth_steps}, B ratio={type_b_ratio*100}% (Process ID: {os.getpid()})")
    
    # Use seeds that do not depend on the sensitivity case. This supplies
    # common random numbers when baseline and perturbed cases are compared.
    label_index = ord(sim_label) - ord('a')
    seed = base_seed + growth_steps * 10000 + label_index
    network = run_simulation(type_b_ratio, topology, params, pressures, dt, a_values, growth_steps, seed=seed)
    
    results = process_results(
        network, type_b_ratio, growth_steps, sim_label, output_dir,
        case_id, parameter, perturbation, baseline_value, perturbed_value,
        seed
    )
    end_time = time.time()
    
    print(f"Completed simulation {sim_label} (growth={growth_steps}) in {end_time - start_time:.2f} seconds")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run normal or sensitivity-mode replicated ZIF-67 simulations."
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help=(
            "Run one sensitivity-analysis case. Without this switch, run "
            "the normal baseline simulations."
        )
    )
    parser.add_argument(
        "--case-id", type=int, default=0,
        help="Sensitivity case number (0 is baseline; requires --sensitivity)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    print(f"Starting ZIF-67 multi-simulations using {mp.cpu_count()} available CPUs...")
    
    zif8 = {
        "n_max": {0: 8, 1: 8},
        "type_accept": {0: [0, 1], 1: [0, 1]},
        "n_directions": {0: 8, 1: 8},
        "directions": [
            [1 if i & (1 << j) else -1 for j in range(3)]
            for i in range(8)
        ],
        "rotation": {0: 1, 1: 1},
        "reverse": {i: 7 - i for i in range(8)}
    }

    topology = zif8
    baseline_params = {
        "wbarrier_a": 11, "dbarrier_a": 2,
        "wbarrier_b": 15, "dbarrier_b": 3,
        "t0": 1,
        "vw_a": 0.23, "vd_a": 0.06,
        "vw_b": 0.19, "vd_b": 0.04
    }

    pressures = np.concatenate([np.linspace(0, 50, num=60000), np.linspace(50, 0, num=60000)])
    dt = 0.4

    # ========== CONFIGURATION ==========
    # Number of simulations per growth step (e.g., a, b, c, d, e, f, g...)
    num_simulations = 5  # Change this to run more/fewer simulations
    
    # Growth steps to test (e.g., 5, 6, 7, 8, 9...)
    growth_steps_list = [8]  # Change this to test different growth steps
    
    # Type B ratio (same for all simulations)
    type_b_ratio = 0.0
    
    # A values
    baseline_a_values = {'AA': 0.95, 'AB': 0.5, 'BA': 0.5, 'BB': 0.01}

    # One-at-a-time sensitivity cases.
    perturbations = (-0.20, -0.10, 0.10, 0.20)
    baseline_values = {
        **baseline_params,
        **{f"A1_{key}": value for key, value in baseline_a_values.items()}
    }
    cases = [('baseline', 0.0)] + [
        (parameter, perturbation)
        for parameter in baseline_values
        for perturbation in perturbations
    ]
    if args.sensitivity:
        if not 0 <= args.case_id < len(cases):
            parser.error(f"--case-id must be between 0 and {len(cases) - 1}")
        parameter, perturbation = cases[args.case_id]
    else:
        if args.case_id != 0:
            parser.error("--case-id requires --sensitivity")
        parameter, perturbation = 'baseline', 0.0
    params = baseline_params.copy()
    a_values = baseline_a_values.copy()
    baseline_value = np.nan
    perturbed_value = np.nan
    if parameter != 'baseline':
        baseline_value = baseline_values[parameter]
        perturbed_value = baseline_value * (1.0 + perturbation)
        if parameter.startswith('A1_'):
            a_values[parameter[3:]] = perturbed_value
        else:
            params[parameter] = perturbed_value
    
    # Base output directory
    case_name = (
        'baseline' if parameter == 'baseline'
        else f"{parameter}_{perturbation:+.0%}"
    )
    base_output_dir = os.path.join(
        args.output_dir,
        f"case_{args.case_id:02d}_{case_name}"
        if args.sensitivity else "normal"
    )
    # ===================================

    # Generate simulation labels (a, b, c, d, e...)
    sim_labels = [chr(97 + i) for i in range(num_simulations)]  # a, b, c, d, e...
    
    # Create all simulation configurations (combinations of growth steps and labels)
    sim_configs = list(itertools.product(growth_steps_list, sim_labels))
    sim_configs = [(growth, label, type_b_ratio) for growth, label in sim_configs]
    
    print(f"Total simulations to run: {len(sim_configs)}")
    print(f"Growth steps: {growth_steps_list}")
    print(f"Simulation labels: {sim_labels}")
    print(f"Type B ratio: {type_b_ratio}")
    if args.sensitivity:
        print(f"Sensitivity case: {args.case_id}/{len(cases) - 1} ({case_name})")
    else:
        print("Mode: normal baseline simulation")
    
    # Number of CPUs to use
    cpu_count = int(os.environ.get('OMP_NUM_THREADS', mp.cpu_count() - 1))
    print(f"Using {cpu_count} CPUs for parallel processing\n")
    
    # Create a pool of worker processes
    pool = mp.Pool(processes=cpu_count)
    
    # Create a partial function with fixed parameters
    sim_func = partial(
        run_parallel_simulation,
        topology=topology,
        params=params,
        pressures=pressures,
        dt=dt,
        a_values=a_values,
        base_output_dir=base_output_dir,
        case_id=args.case_id,
        parameter=parameter,
        perturbation=perturbation,
        baseline_value=baseline_value,
        perturbed_value=perturbed_value,
        base_seed=args.seed
    )
    
    # Run simulations in parallel
    start_time = time.time()
    results = pool.map(sim_func, sim_configs)
    pool.close()
    pool.join()
    end_time = time.time()
    
    print(f"\nAll simulations completed in {end_time - start_time:.2f} seconds")
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame([
        {
            'Mode': 'sensitivity' if args.sensitivity else 'normal',
            'Case_ID': r['case_id'] if args.sensitivity else np.nan,
            'Parameter': r['parameter'],
            'Perturbation_Percent': r['perturbation_percent'],
            'Baseline_Value': r['baseline_value'],
            'Perturbed_Value': r['perturbed_value'],
            'Random_Seed': r['random_seed'],
            'Simulation_Label': r['simulation_label'],
            'Growth_Steps': r['growth_steps'],
            'Type_B_Ratio': r['type_b_ratio'],
            'Hysteresis': r['hysteresis'],
            'Saturation_Step': r['saturation_step'],
            'Saturation_Pressure': r['saturation_pressure'],
            'Maximum_Filling': r['maximum_filling'],
            'Final_Filling': r['final_filling'],
            'Total_Nodes': r['total_nodes'],
            'Nodes_With_8_Neighbors': r['nodes_with_8_neighbors'],
            'Nodes_With_8_Neighbors_Percentage': r['nodes_with_8_neighbors_percentage'],
            'Surface_Nodes': r['surface_nodes'],
            'Surface_Nodes_Percentage': r['surface_nodes_percentage'],
            'A_Type_Surface_Nodes': r['surface_distribution']['a_type_surface'],
            'B_Type_Surface_Nodes': r['surface_distribution']['b_type_surface'],
            'A_Type_Surface_Percentage': r['surface_distribution']['a_type_percentage'],
            'B_Type_Surface_Percentage': r['surface_distribution']['b_type_percentage']
        }
        for r in results
    ])
    
    # Save overall results
    os.makedirs(base_output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(base_output_dir, 'overall_results.csv'), index=False)
    print(f"\nOverall results saved to '{base_output_dir}/overall_results.csv'")
    
    # Print summary statistics by growth step
    print("\n" + "="*60)
    print("SUMMARY BY GROWTH STEP")
    print("="*60)
    for growth in growth_steps_list:
        growth_data = results_df[results_df['Growth_Steps'] == growth]
        print(f"\nGrowth Steps: {growth}")
        print(f"  Average Hysteresis: {growth_data['Hysteresis'].mean():.4f} ± {growth_data['Hysteresis'].std():.4f}")
        print(f"  Average Total Nodes: {growth_data['Total_Nodes'].mean():.1f} ± {growth_data['Total_Nodes'].std():.1f}")
        print(f"  Average Surface Nodes: {growth_data['Surface_Nodes'].mean():.1f} ± {growth_data['Surface_Nodes'].std():.1f}")
    
    print("\nAll simulations completed successfully!")
