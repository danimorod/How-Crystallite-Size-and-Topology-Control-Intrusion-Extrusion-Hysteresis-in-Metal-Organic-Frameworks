import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import sys
import logging
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - Nucleation Regime Classification (Sear 2013 Framework)
# ============================================================================
BETA_THRESHOLD_RATELSS = 0.7      # Beta < 0.7: Rate-less nucleation
BETA_THRESHOLD_INTERMEDIATE = 0.9  # 0.7 <= Beta < 0.9: Intermediate
BETA_THRESHOLD_CLASSICAL_MAX = 1.1  # 0.9 <= Beta <= 1.1: Classical
MIN_NUCLEATION_EVENTS = 10         # Minimum events for reliable fitting
SURVIVAL_CUTOFF = 0.01            # Filter noise in tail of survival curve
BETA_INIT_GUESSES = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]  # Initial guesses for fitting
BETA_BOUNDS_MIN = 0.1              # Minimum beta value in fits
BETA_BOUNDS_MAX = 5.0              # Maximum beta value in fits

# ZIF-67 SOD topology definition (similar to ZIF-8, single-cage structure)
def get_topology_zif67_sod():
    """
    ZIF-67 SOD topology definition
    Cubic cage structure (SOD - Sodalite topology)
    Each node has 8 coordination sites (cubic arrangement)
    """
    return {
        "n_max": {0: 8},
        "type_accept": {0: [0]},
        "n_directions": {0: 8},
        "directions": [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ],
        "rotation": {0: 1},
        "reverse": {
            0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0
        }
    }

zif67_sod = get_topology_zif67_sod()
nmax = zif67_sod["n_max"][0]
n_directions = len(zif67_sod["directions"])

class Node:
    def __init__(self, x, y, z, n_number, cage_type=0):
        self.x = x
        self.y = y
        self.z = z
        self.n_number = n_number
        self.n_list = []
        self.state = 1  # Start as liquid (1) for extrusion simulation
        self.wet_neighbours = {'A': 0, 'B': 0}  # Track neighbors by type
        self.surface = False
        self.is_nucleation_site = False
        self.cage_type = cage_type  # 0 for A-type, 1 for B-type

    def add_neighbour(self, node):
        if node not in self.n_list:
            self.n_list.append(node)
            node.n_list.append(self)

    def surface_node(self):
        self.state = 1  # Surface nodes are liquid
        self.surface = True
        for node in self.n_list:
            node_type = 'A' if self.cage_type == 0 else 'B'
            node.wet_neighbours[node_type] += 1

    def update_state(self, change):
        if change and not self.surface:
            dif = 1 - self.state * 2  # if state==1 -> dif=-1 ; if state==0 -> dif=1
            self.state = 1 - self.state
            cage_type_str = 'A' if self.cage_type == 0 else 'B'
            for n in self.n_list:
                n.wet_neighbours[cage_type_str] += dif

class Network:
    def __init__(self, topology, params, A1_values):
        self.topology = topology
        self.params = params
        self.A1_values = A1_values  # Dict with 'AA', 'AB', 'BA', 'BB' keys
        self.nodes = {}
        self.table_rates = None
        self.pressure = None
        self.first_nucleation_time = None
        self.nucleation_sites = []
        self.nucleating_cluster_size = None

    def build_network(self, growth_level, type_b_ratio=0.0):
        queue = deque([(0, 0, 0, 0)])
        visited = {(0, 0, 0)}
        level = 0

        while queue and level <= growth_level:
            level_size = len(queue)
            for _ in range(level_size):
                x, y, z, n_number = queue.popleft()
                # Randomly assign cage type (0=A, 1=B)
                cage_type = 1 if np.random.rand() < type_b_ratio else 0
                node = Node(x, y, z, n_number, cage_type=cage_type)
                self.nodes[(x, y, z)] = node

                for direction in self.topology["directions"]:
                    nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]
                    if (nx, ny, nz) not in visited:
                        visited.add((nx, ny, nz))
                        queue.append((nx, ny, nz, len(visited) - 1))

            level += 1

        for (x, y, z), node in self.nodes.items():
            for direction in self.topology["directions"]:
                nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]
                if (nx, ny, nz) in self.nodes:
                    neighbor = self.nodes[(nx, ny, nz)]
                    if neighbor not in node.n_list:
                        node.add_neighbour(neighbor)

    def get_surface(self):
        surface = []
        for node in self.nodes.values():
            if len(node.n_list) < self.topology["n_max"][0]:
                surface.append(node)
        return surface

    def complex_A1(self, n_A, n_B, cage_type):
        """Calculate interfacial tension parameter with cage-type dependent interactions.
        
        Args:
            n_A: Number of filled A-type neighbors
            n_B: Number of filled B-type neighbors
            cage_type: Type of current cage (0=A, 1=B)
        """
        nmax = self.topology["n_max"][0]
        # Complex interaction: A-A, A-B, B-A, B-B
        A1 = (n_A * (n_A * self.A1_values['AA'] + n_B * self.A1_values['AB']) + 
              n_B * (n_A * self.A1_values['BA'] + n_B * self.A1_values['BB'])) / (nmax ** 2)
        return A1

    def calculate_rates_at_pressure(self, pressure):
        """Calculate wetting and dewetting rates at a given pressure.
        
        Args:
            pressure: Applied pressure in bar
            
        Returns:
            rates: 4D array of shape (2, n_max+1, n_max+1, 2) with rates for each cage type
                   rates[cage_type, n, n_a, 0/1] = wetting/dewetting rate
        """
        nmax = self.topology["n_max"][0]
        wbarrier_a = self.params["wbarrier_a"]
        dbarrier_a = self.params["dbarrier_a"]
        wbarrier_b = self.params["wbarrier_b"]
        dbarrier_b = self.params["dbarrier_b"]
        vw_a = self.params["vw_a"]
        vd_a = self.params["vd_a"]
        vw_b = self.params["vw_b"]
        vd_b = self.params["vd_b"]
        t0 = self.params["t0"]

        rates = np.zeros((2, nmax + 1, nmax + 1, 2))

        for cage_type in [0, 1]:
            for n in range(nmax + 1):
                for n_a in range(n + 1):
                    n_b = n - n_a
                    
                    A1 = self.complex_A1(n_a, n_b, cage_type)
                    
                    barrier_w = wbarrier_a if cage_type == 0 else wbarrier_b
                    barrier_d = dbarrier_a if cage_type == 0 else dbarrier_b
                    vw = vw_a if cage_type == 0 else vw_b
                    vd = vd_a if cage_type == 0 else vd_b
                    
                    omegaw = max(barrier_w - pressure * vw + (nmax / 2 - n) * A1, 0)
                    omegad = max(barrier_d + pressure * vd + (n - nmax / 2) * A1, 0)
                    
                    rates[cage_type, n, n_a, 0] = (1.0 / t0) * np.exp(-omegaw)
                    rates[cage_type, n, n_a, 1] = (1.0 / t0) * np.exp(-omegad)

        return rates

    def initialize_simulation(self, pressures, dt, func, params):
        self.pressure = func(pressures, params)
        self.table_rates = self.calculate_rates_at_pressure(self.pressure)
        self.first_nucleation_time = None
        self.nucleation_sites = []
        self.nucleating_cluster_size = None

        for n in self.nodes.values():
            n.state = 1  # All nodes start as liquid
            n.wet_neighbours = {'A': 0, 'B': 0}
            n.is_nucleation_site = False
            # Count neighbors by type
            for neighbor in n.n_list:
                neighbor_type = 'A' if neighbor.cage_type == 0 else 'B'
                n.wet_neighbours[neighbor_type] += 1

        self.surface_nodes = self.get_surface()
        for n in self.surface_nodes:
            n.surface_node()

        self.update_rates()

    def update_rates(self):
        nmax = self.topology["n_max"][0]
        
        for n in self.nodes.values():
            cage_type = n.cage_type
            n_a = min(n.wet_neighbours['A'], nmax)
            n_b = min(n.wet_neighbours['B'], nmax)
            n_total = min(n_a + n_b, nmax)
            
            # Get rates from precomputed table
            if n_total <= nmax and n_a <= nmax:
                n.fill_rate = self.table_rates[cage_type, n_total, n_a, 0]
                n.empty_rate = self.table_rates[cage_type, n_total, n_a, 1]
            else:
                n.fill_rate = 0
                n.empty_rate = 0

    def dry_clusters(self):
        """Return all connected components formed by dry (gas-state) nodes."""
        unvisited = {node for node in self.nodes.values() if node.state == 0}
        clusters = []

        while unvisited:
            start = unvisited.pop()
            cluster = [start]
            stack = [start]

            while stack:
                node = stack.pop()
                for neighbor in node.n_list:
                    if neighbor.state == 0 and neighbor in unvisited:
                        unvisited.remove(neighbor)
                        cluster.append(neighbor)
                        stack.append(neighbor)

            clusters.append(cluster)

        return clusters

    def detect_nucleation(self, time_step, n_c):
        """Record the first dry component whose size reaches the critical size."""
        if not isinstance(n_c, (int, np.integer)) or n_c < 1:
            raise ValueError("n_c must be a positive integer")

        clusters = self.dry_clusters()
        largest_cluster_size = max((len(cluster) for cluster in clusters), default=0)

        if self.first_nucleation_time is None and largest_cluster_size >= n_c:
            nucleating_cluster = max(clusters, key=len)
            self.first_nucleation_time = time_step
            self.nucleating_cluster_size = len(nucleating_cluster)
            self.nucleation_sites = [(time_step, node) for node in nucleating_cluster]
            for node in nucleating_cluster:
                node.is_nucleation_site = True

        return largest_cluster_size

    def one_step(self, dt, time_step, spontaneous_base_prob=1e-4, volume_scaling=False, n_c=1):
        """Execute one simulation timestep: wetting/dewetting and spontaneous nucleation.
        
        Args:
            dt: Timestep size
            time_step: Current time step number
            spontaneous_base_prob: Base probability for spontaneous nucleation (default: 1e-4)
            volume_scaling: If True, scale spontaneous probability by 1/N_internal (classical); 
                          If False, keep constant per-node (natural)
            n_c: Critical connected dry-cluster size used to define nucleation
                          
        Returns:
            Dict with statistics about growth attempts, spontaneous checks, and nucleation attempts
        """
        n_nodes = len(self.nodes)
        n_internal = sum(1 for n in self.nodes.values() if not n.surface)
        
        if volume_scaling:
            # Optional: use 1/N scaling for classical Sear comparison
            scale = 1.0 / max(1.0, n_internal)
        else:
            # Natural scaling - keep per-node probability constant
            scale = 1.0
        
        p_sp = spontaneous_base_prob * scale

        states = np.array([n.state for n in self.nodes.values()])
        rates = np.array([[n.empty_rate, n.fill_rate] for n in self.nodes.values()])

        randoms = np.random.rand(n_nodes)
        prop_acceptance = 1 - np.exp(- (1 - states) * rates[:, 0] * dt - states * rates[:, 1] * dt)
        change = randoms < prop_acceptance

        gas_neighbors = {}
        for i, node in enumerate(self.nodes.values()):
            gas_neighbors[i] = any(neighbor.state == 0 for neighbor in node.n_list)

        growth_attempts = 0
        spontaneous_checks = 0
        nucleation_attempts = 0

        for i, node in enumerate(self.nodes.values()):
            if node.state == 1 and not node.surface:  # Liquid nodes only
                has_gas_neighbor = gas_neighbors[i]
                
                if has_gas_neighbor:
                    # Growth attempt (liquid next to gas)
                    growth_attempts += 1
                else:
                    # Spontaneous nucleation attempt (no gas neighbors)
                    spontaneous_checks += 1
                    if randoms[i] < p_sp:
                        change[i] = True
                        nucleation_attempts += 1

        for i, node in enumerate(self.nodes.values()):
            node.update_state(change[i])

        self.update_rates()
        largest_cluster_size = self.detect_nucleation(time_step, n_c)
        
        return {
            'growth_attempts': growth_attempts,
            'spontaneous_checks': spontaneous_checks,
            'nucleation_attempts': nucleation_attempts,
            'largest_dry_cluster': largest_cluster_size
        }

def pressure_function(pressures, params):
    return params["pressure"]

def calculate_survival_function(first_times, max_time=None):
    """
    Calculate Kaplan-Meier survival function for first nucleation times.
    More robust than simple counting method.
    
    Args:
        first_times: Array of first nucleation times from multiple runs
        max_time: Maximum time to consider (default: max of first_times)
        
    Returns:
        Tuple of (times, survival_probabilities) where survival gives P(T > t)
    """
    if len(first_times) == 0:
        return [], []
    
    first_times = np.sort(first_times)
    if max_time is None:
        max_time = int(np.max(first_times))
    
    times = []
    survival = []
    n_at_risk = len(first_times)
    
    for t in range(max_time + 1):
        # Number of events at this time
        n_events = np.sum(first_times == t)
        
        if n_events > 0:
            times.append(t)
            # Kaplan-Meier estimator
            survival_prob = 1 - (n_events / n_at_risk) if n_at_risk > 0 else 0
            if len(survival) > 0:
                survival.append(survival[-1] * survival_prob)
            else:
                survival.append(survival_prob)
            n_at_risk -= n_events
        elif len(times) > 0 and n_at_risk > 0:
            # No events, but we still have data points at risk
            times.append(t)
            survival.append(survival[-1] if len(survival) > 0 else 1.0)
    
    # Filter out very small survival probabilities (noise in tail)
    valid_idx = [i for i, s in enumerate(survival) if s > SURVIVAL_CUTOFF]
    if len(valid_idx) > 0:
        times = [times[i] for i in valid_idx]
        survival = [survival[i] for i in valid_idx]
    
    return times, survival

def fit_stretched_exponential(times, survival_fraction):
    """
    Fit stretched exponential: S(t) = exp(-(t/tau)^beta)
    According to Sear (2013):
    - beta ≈ 1: Classical nucleation (exponential)
    - beta < 1: Rate-less nucleation (sub-exponential)
    - beta > 1: Super-exponential (rare)
    
    Args:
        times: Array of time points
        survival_fraction: Survival probabilities at each time
        
    Returns:
        Tuple of (tau, beta, r2, rmse) or (None, None, None, None) if fit fails
    """
    def stretched_exp(t, tau, beta):
        return np.exp(- (t / tau) ** beta)
    
    if len(times) < MIN_NUCLEATION_EVENTS:
        logger.warning(f"Insufficient data for fitting: {len(times)} points (need >= {MIN_NUCLEATION_EVENTS})")
        return None, None, None, None
    
    # Ensure inputs are valid
    if not (isinstance(times, (list, np.ndarray)) and isinstance(survival_fraction, (list, np.ndarray))):
        logger.error("Invalid input types for fit_stretched_exponential")
        return None, None, None, None
        
    times = np.array(times, dtype=float)
    survival_fraction = np.array(survival_fraction, dtype=float)
    
    # Check for valid data
    if np.any(np.isnan(times)) or np.any(np.isnan(survival_fraction)):
        logger.error("NaN values found in fit data")
        return None, None, None, None
    
    # Try multiple initial guesses for robustness
    best_fit = None
    best_r2 = -np.inf
    best_rmse = np.inf
    
    # More comprehensive initial guesses
    for beta_init in BETA_INIT_GUESSES:
        try:
            # Better tau initialization
            tau_init = np.median(times)
            
            popt, pcov = curve_fit(
                stretched_exp, 
                times, 
                survival_fraction,
                p0=[tau_init, beta_init],
                bounds=([0.1, BETA_BOUNDS_MIN], [np.inf, BETA_BOUNDS_MAX]),
                maxfev=10000
            )
            
            tau_fit, beta_fit = popt
            
            # Calculate goodness of fit metrics
            y_pred = stretched_exp(times, *popt)
            
            # R²
            ss_res = np.sum((survival_fraction - y_pred) ** 2)
            ss_tot = np.sum((survival_fraction - np.mean(survival_fraction)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # RMSE
            rmse = np.sqrt(np.mean((survival_fraction - y_pred) ** 2))
            
            # Prefer fits with better R² and lower RMSE
            if r2 > best_r2 or (abs(r2 - best_r2) < 0.01 and rmse < best_rmse):
                best_r2 = r2
                best_rmse = rmse
                best_fit = (tau_fit, beta_fit, r2, rmse)
                
        except (RuntimeError, ValueError, OverflowError) as e:
            logger.debug(f"Fit failed with beta_init={beta_init}: {str(e)}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error in curve fitting: {str(e)}")
            continue
    
    if best_fit is not None:
        return best_fit
    return None, None, None, None

def run_single_realization(args):
    topology, params, pressure, dt, a_values, growth_level, max_steps, spontaneous_base_prob, volume_scaling, type_b_ratio, n_c, seed = args
    
    if seed is not None:
        np.random.seed(seed)
    
    net = Network(topology, params, a_values)
    net.build_network(growth_level, type_b_ratio=type_b_ratio)
    net.initialize_simulation([pressure], dt, pressure_function, {"pressure": pressure})
    
    n_nodes = len(net.nodes)
    time_step = 0
    frac_gas = 0.0
    
    total_growth = 0
    total_spontaneous = 0
    total_nucleation = 0
    
    while time_step < max_steps and frac_gas < 0.5:
        stats = net.one_step(dt, time_step, spontaneous_base_prob, volume_scaling, n_c)
        total_growth += stats['growth_attempts']
        total_spontaneous += stats['spontaneous_checks']
        total_nucleation += stats['nucleation_attempts']
        
        time_step += 1
        n_gas = sum(1 for n in net.nodes.values() if n.state == 0)
        frac_gas = n_gas / n_nodes
    
    return {
        'first_nucleation_time': net.first_nucleation_time,
        'n_nodes': n_nodes,
        'final_time': time_step,
        'final_frac_gas': frac_gas,
        'n_nucleation_sites': len(net.nucleation_sites),
        'nucleating_cluster_size': net.nucleating_cluster_size,
        'growth_attempts': total_growth,
        'spontaneous_checks': total_spontaneous,
        'nucleation_attempts': total_nucleation
    }

def volume_scaling_study(topology, params, pressure, dt, a_values,
                         growth_levels, n_runs_per_level=50, max_steps=10000,
                         spontaneous_base_prob=1e-4, volume_scaling=False, type_b_ratio=0.0,
                         n_c=1):
    
    results_list = []
    
    for g in growth_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Growth level {g}")
        logger.info(f"{'='*60}")
        
        # Generate unique seeds for reproducibility
        seeds = [int((time.time() * 1000) % 2**31) ^ (i * 9301 + 49297) for i in range(n_runs_per_level)]
        
        args_list = [
            (topology, params, pressure, dt, a_values, g, max_steps, 
             spontaneous_base_prob, volume_scaling, type_b_ratio, n_c, seeds[i])
            for i in range(n_runs_per_level)
        ]
        
        # Use multiprocessing for smaller networks, sequential for larger
        if g < 5:
            with Pool(processes=min(cpu_count(), n_runs_per_level)) as pool:
                results = pool.map(run_single_realization, args_list)
        else:
            results = [run_single_realization(args) for args in args_list]
        
        # Analyze results
        first_times = [r['first_nucleation_time'] for r in results if r['first_nucleation_time'] is not None]
        n_nucleated = len(first_times)
        n_runs = len(results)
        nucleation_rate = n_nucleated / n_runs
        
        n_nodes = results[0]['n_nodes']
        # Count nodes that are not surface nodes (correctly identify internal vs surface)
        # Surface nodes = nodes with fewer neighbors than nmax
        nmax = topology['n_max'][0]
        net_temp = Network(topology, params, a_values)
        net_temp.build_network(g, type_b_ratio=type_b_ratio)
        n_internal = sum(1 for n in net_temp.nodes.values() if len(n.n_list) == nmax)
        
        # Calculate effective probability
        if volume_scaling:
            effective_p_sp = spontaneous_base_prob / max(1.0, n_internal)
        else:
            effective_p_sp = spontaneous_base_prob
        
        expected_events = n_internal * max_steps * effective_p_sp
        
        print(f"Nodes: {n_nodes}")
        print(f"Nucleation success rate: {n_nucleated}/{n_runs} ({nucleation_rate:.2%})")
        print(f"Effective p_sp: {effective_p_sp:.2e}")
        print(f"Expected events per run: {expected_events:.2f}")
        
        if n_nucleated > 0:
            total_growth = sum(r['growth_attempts'] for r in results)
            total_spontaneous = sum(r['spontaneous_checks'] for r in results)
            total_nucleation = sum(r['nucleation_attempts'] for r in results)
            
            print(f"Total growth attempts: {total_growth}")
            print(f"Total spontaneous checks: {total_spontaneous}")
            print(f"Total nucleation attempts: {total_nucleation}")
            print(f"Nucleation success rate: {total_nucleation}/{total_spontaneous} ({total_nucleation/max(1,total_spontaneous):.2%})")
        
        if n_nucleated >= MIN_NUCLEATION_EVENTS:
            mean_first = np.mean(first_times)
            std_first = np.std(first_times)
            median_first = np.median(first_times)
            
            # Calculate survival function using Kaplan-Meier estimator
            times_for_fit, survival = calculate_survival_function(first_times)
            
            # Fit stretched exponential to survival function
            tau, beta, r2, rmse = fit_stretched_exponential(times_for_fit, survival)
            
            # Sear's criterion for rate-less nucleation
            # If beta < 0.7, likely rate-less; if beta ≈ 1, classical
            nucleation_regime = 'Unknown'
            if beta is not None:
                if beta < BETA_THRESHOLD_RATELSS:
                    nucleation_regime = 'Rate-less (Sear)'
                elif BETA_THRESHOLD_RATELSS <= beta < BETA_THRESHOLD_INTERMEDIATE:
                    nucleation_regime = 'Intermediate'
                elif BETA_THRESHOLD_INTERMEDIATE <= beta <= BETA_THRESHOLD_CLASSICAL_MAX:
                    nucleation_regime = 'Classical'
                else:
                    nucleation_regime = 'Super-exponential'
            
            results_list.append({
                'growth_level': g,
                'n_nodes': n_nodes,
                'n_c': n_c,
                'beta': beta,
                'tau': tau,
                'r2': r2,
                'rmse': rmse if rmse is not None else np.nan,
                't_median': median_first,
                'mean_first_time': mean_first,
                'std_first_time': std_first,
                'n_nucleated': n_nucleated,
                'nucleation_rate': nucleation_rate,
                'nucleation_regime': nucleation_regime
            })
            
            print(f"Mean first nucleation time: {mean_first:.2f} ± {std_first:.2f}")
            print(f"Median: {median_first:.2f}")
            if beta is not None:
                print(f"Stretched exponential: tau={tau:.2f}, beta={beta:.2f}, R²={r2:.4f}")
                print(f"Nucleation regime: {nucleation_regime}")
                print(f"RMSE: {rmse:.4f}")
        else:
            print(f"WARNING: Only {n_nucleated} nucleation events observed!")
            print(f"Consider:")
            print(f"  - Increasing spontaneous_base_prob (current: {spontaneous_base_prob:.2e})")
            print(f"  - Lowering pressure (current: {pressure})")
            print(f"  - Increasing max_steps (current: {max_steps})")
            
            results_list.append({
                'growth_level': g,
                'n_nodes': n_nodes,
                'n_c': n_c,
                'beta': None,
                'tau': None,
                'r2': None,
                'rmse': np.nan,
                't_median': max_steps if n_nucleated == 0 else np.median(first_times),
                'mean_first_time': 0.0 if n_nucleated == 0 else np.mean(first_times),
                'std_first_time': 0.0 if n_nucleated == 0 else np.std(first_times),
                'n_nucleated': n_nucleated,
                'nucleation_rate': nucleation_rate,
                'nucleation_regime': 'Insufficient data'
            })
    
    return pd.DataFrame(results_list)

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    
    # ZIF-67 Parameters (SOD topology with two cage types)
    # From zif67_sim.py
    params = {
        "wbarrier_a": 11.0,    # Wetting barrier for A-type cage
        "dbarrier_a": 2.0,     # Dewetting barrier for A-type cage
        "wbarrier_b": 15.0,    # Wetting barrier for B-type cage
        "dbarrier_b": 3.0,     # Dewetting barrier for B-type cage
        "t0": 1.0,             # Time scale
        "vw_a": 0.23,          # Molar volume for wetting (A-type)
        "vd_a": 0.06,          # Molar volume for dewetting (A-type)
        "vw_b": 0.19,          # Molar volume for wetting (B-type)
        "vd_b": 0.04           # Molar volume for dewetting (B-type)
    }
    
    # Pressure sweep - test multiple pressures
    pressure_range = [25.0, 26.0, 27.0]  # Multiple pressures to explore
    dt = 0.1                  # Timestep (from zif67_sim.py)
    a_values = {'AA': 0.95, 'AB': 0.5, 'BA': 0.5, 'BB': 0.01}  # Interfacial tension parameters
    growth_levels = [8]  # Test different network sizes
    n_runs = 100
    max_steps = 50000
    spontaneous_base_prob = 1e-4  # Base probability for spontaneous nucleation
    volume_scaling = False  # Natural scaling - no artificial modification
    type_b_ratio = 0.0      # Fraction of B-type cages (0.0 = all A-type, 1.0 = all B-type)
    n_c = 5                 # Critical connected dry-cluster size for nucleation
    
    print("="*70)
    print("ZIF-67 SOD Dry Bubble Nucleation - Pressure Sweep Study")
    print("="*70)
    print(f"Topology: SOD (Sodalite - cubic cage structure)")
    print(f"Pressure range: {pressure_range} bar")
    print(f"Spontaneous base probability: {spontaneous_base_prob:.2e}")
    print(f"Volume scaling: {volume_scaling}")
    print(f"Scaling type: {'Natural (constant per-node rate)' if not volume_scaling else 'Classical (1/N)'}")
    print(f"Growth levels: {growth_levels}")
    print(f"Runs per level: {n_runs}")
    print(f"Max steps: {max_steps}")
    print(f"Critical dry-cluster size n_c: {n_c}")
    print("="*70)
    
    # Storage for all results
    all_results = []
    pressure_scaling_exponents = []
    
    # Run simulation for each pressure
    for pressure in pressure_range:
        print(f"\n{'#'*70}")
        print(f"# PRESSURE = {pressure} bar")
        print(f"{'#'*70}")
        
        df_pressure = volume_scaling_study(
            zif67_sod, params, pressure, dt, a_values,
            growth_levels=growth_levels,
            n_runs_per_level=n_runs,
            max_steps=max_steps,
            spontaneous_base_prob=spontaneous_base_prob,
            volume_scaling=volume_scaling,
            type_b_ratio=type_b_ratio,
            n_c=n_c
        )
        
        # Add pressure column
        df_pressure['pressure'] = pressure
        all_results.append(df_pressure)
        
        # Calculate scaling exponent for this pressure
        df_clean = df_pressure.dropna(subset=['mean_first_time'])
        df_clean = df_clean[df_clean['mean_first_time'] > 0]
        
        if len(df_clean) >= 2:
            x = np.log(df_clean['n_nodes'].values)
            y = np.log(df_clean['mean_first_time'].values)
            coeffs = np.polyfit(x, y, 1)
            alpha = coeffs[0]
            
            # Calculate R²
            y_fit = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            pressure_scaling_exponents.append({
                'pressure': pressure,
                'alpha': alpha,
                'r2': r2,
                'n_points': len(df_clean)
            })
            
            print(f"\nPressure {pressure}: α = {alpha:.3f}, R² = {r2:.4f}")
    
    # Combine all results
    df = pd.concat(all_results, ignore_index=True)
    
    print("\n" + "="*70)
    print("COMBINED RESULTS SUMMARY (All Pressures)")
    print("="*70)
    print(df.to_string(index=False))
    
    # Analysis across pressures
    print("\n" + "="*70)
    print("PRESSURE-DEPENDENT ANALYSIS")
    print("="*70)
    
    if len(pressure_scaling_exponents) > 0:
        df_pressure_exponents = pd.DataFrame(pressure_scaling_exponents)
        print("\nScaling Exponent vs Pressure:")
        print(df_pressure_exponents.to_string(index=False))
        
        # Plot pressure dependence
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scaling exponent vs pressure
        ax = axes[0, 0]
        ax.plot(df_pressure_exponents['pressure'], df_pressure_exponents['alpha'], 
               'o-', markersize=12, linewidth=2.5, color='darkblue', label='α(P)')
        ax.axhline(y=-1.0, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Classical (α=-1)')
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Rate-less (α=0)')
        ax.fill_between(df_pressure_exponents['pressure'], -1.0, 0.0, alpha=0.1, color='orange')
        ax.set_xlabel('Pressure (bar)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Scaling Exponent α', fontsize=13, fontweight='bold')
        ax.set_title('Volume Scaling vs Pressure (ZIF-67 SOD)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. Nucleation rate vs pressure (at fixed size)
        ax = axes[0, 1]
        for g in growth_levels:
            data_at_level = df[df['growth_level'] == g].groupby('pressure').agg({
                'nucleation_rate': 'mean',
                'n_nodes': 'first'
            }).reset_index()
            ax.plot(data_at_level['pressure'], data_at_level['nucleation_rate'], 
                   'o-', markersize=10, linewidth=2, label=f'Level {g} ({data_at_level["n_nodes"].iloc[0]:.0f} nodes)')
        ax.set_xlabel('Pressure (bar)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Nucleation Success Rate', fontsize=13, fontweight='bold')
        ax.set_title('Nucleation Rate vs Pressure (at fixed sizes)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Mean first nucleation time vs pressure
        ax = axes[1, 0]
        for g in growth_levels:
            data_at_level = df[df['growth_level'] == g].groupby('pressure').agg({
                'mean_first_time': 'mean',
                'n_nodes': 'first'
            }).reset_index()
            ax.plot(data_at_level['pressure'], data_at_level['mean_first_time'], 
                   's-', markersize=10, linewidth=2, label=f'Level {g} ({data_at_level["n_nodes"].iloc[0]:.0f} nodes)')
        ax.set_xlabel('Pressure (bar)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean First Nucleation Time', fontsize=13, fontweight='bold')
        ax.set_title('Nucleation Time vs Pressure (linear axes)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. Beta parameter trends
        ax = axes[1, 1]
        for regime in ['Classical', 'Intermediate', 'Rate-less (Sear)', 'Super-exponential']:
            data_regime = df[df['nucleation_regime'] == regime].groupby('pressure').agg({
                'beta': 'mean',
                'n_nodes': 'first'
            }).reset_index()
            if len(data_regime) > 0:
                ax.plot(data_regime['pressure'], data_regime['beta'], 
                       'o-', markersize=10, linewidth=2, label=regime)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='β=1')
        ax.axhline(y=0.7, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='β=0.7')
        ax.set_xlabel('Pressure (bar)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Stretched Exponential β', fontsize=13, fontweight='bold')
        ax.set_title('Nucleation Regime Dependence on Pressure', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('zif67_sod_pressure_sweep_analysis.png', dpi=300, bbox_inches='tight')
        print("\nPressure sweep plot saved to: zif67_sod_pressure_sweep_analysis.png")
        plt.show()
        
        # Save pressure sweep data
        df_pressure_exponents.to_csv('zif67_sod_pressure_scaling_exponents.csv', index=False)
        print(f"\nPressure scaling exponents saved to: zif67_sod_pressure_scaling_exponents.csv")
    
    # Save combined results
    df.to_csv('zif67_sod_sear_pressure_sweep_results.csv', index=False)
    print(f"Combined results saved to: zif67_sod_sear_pressure_sweep_results.csv")
    
    # Summary statistics by pressure
    print("\n" + "="*70)
    print("SUMMARY STATISTICS BY PRESSURE")
    print("="*70)
    
    for pressure in pressure_range:
        df_p = df[df['pressure'] == pressure]
        nucleation_rate = df_p['nucleation_rate'].mean()
        mean_beta = df_p['beta'].mean()
        
        print(f"\nPressure {pressure} bar:")
        print(f"  Average nucleation success rate: {nucleation_rate:.2%}")
        print(f"  Mean β parameter: {mean_beta:.3f}")
        print(f"  Regime distribution:")
        regime_dist = df_p['nucleation_regime'].value_counts()
        for regime, count in regime_dist.items():
            print(f"    {regime}: {count}")
