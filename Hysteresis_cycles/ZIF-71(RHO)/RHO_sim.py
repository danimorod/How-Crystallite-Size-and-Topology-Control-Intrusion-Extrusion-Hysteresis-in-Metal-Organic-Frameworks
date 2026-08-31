import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
import pandas as pd
import argparse

class Node:
    def __init__(self, ntype, position, topology):
        self.ntype = ntype
        self.position = np.array(position)
        self.free_directions = np.arange(topology["n_directions"][ntype])
        self.topology = topology
        self.n_number = 0
        self.n_list = []
        self.directions = []
        self.nid = None
        self.state = 0
        self.wet_neighbours = 0
        self.surface = False
        self.hash = str(np.round(self.position, 3))
        self.n_max = topology["n_max"][ntype]

    def update_id(self, nid):
        self.nid = nid

    def add_neigh(self, node, direction):
        if node not in self.n_list:
            self.n_list.append(node)
            self.directions.append(direction)
            self.n_number += 1
            self.free_directions = np.setdiff1d(self.free_directions, [direction])

        if self.n_number > self.n_max:
            raise ValueError("Maximum number of neighbors exceeded")

    def merge(self, node):
        for neigh in node.n_list:
            if neigh not in self.n_list:
                self.n_list.append(neigh)
                neigh.n_list = [n if n != node else self for n in neigh.n_list]

        self.free_directions = np.union1d(self.free_directions, node.free_directions)
        self.wet_neighbours = max(self.wet_neighbours, node.wet_neighbours)
        self.surface = self.surface or node.surface
        self.n_number = len(self.n_list)

    def surface_node(self):
        self.state = 1
        self.surface = True
        for node in self.n_list:
            node.wet_neighbours += 1

    def update_state(self, change):
        if change and not self.surface:
            dif = 1 - self.state * 2
            self.state = 1 - self.state
            for n in self.n_list:
                n.wet_neighbours += dif

class Network:
    def __init__(self, topology, A1_value_hex, A1_value_oct):
        self.lastid = 1
        self.nodes = {}
        self.topology = topology
        self.states = None
        self.rates = None
        self.table_rates = None
        self.rate_params = None
        self.dt = None
        self.average_filling = []
        self.pressures = None
        self.A1_value_hex = A1_value_hex
        self.A1_value_oct = A1_value_oct
        self.surface_nodes = []

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
                    new = Node(ntype, nposition, self.topology)
                    ndir = self.topology["reverse"][ndirection]
                    new.add_neigh(node, ndir)
                    newnetwork["nodes"].append([new, node, ndirection])

            for nodepair in newnetwork["nodes"]:
                node, old, direction = nodepair
                position = node.position
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

    def complex_A1(self, n, direction):
        nmax = self.topology["n_max"][0]
        is_octagonal = direction >= 8
        A_AA = self.A1_value_oct if is_octagonal else self.A1_value_hex
        return (n * n * A_AA) / (nmax ** 2)

    def default_function(self, pressures, params):
        nmax = self.topology["n_max"][0]
        wbarrier, dbarrier, vw, vd, t0 = params.values()

        prates = []
        for p in pressures:
            rates = []
            for n in range(nmax + 1):
                direction_rates = []
                for direction in range(self.topology["n_directions"][0]):
                    A1 = self.complex_A1(n, direction)

                    omegaw = max(wbarrier - p * vw + (nmax/2 - n) * A1, 0)
                    omegad = max(dbarrier + p * vd + (n - nmax/2) * A1, 0)

                    w = 1 / t0 * np.exp(-omegaw)
                    d = 1 / t0 * np.exp(-omegad)
                    direction_rates.append([w, d])
                rates.append(direction_rates)
            prates.append(rates)

        return np.array(prates)

    def tabulate_rates(self, func, pressures, params):
        # Store the effective rate averaged over the 8 hexagonal and 6
        # octagonal directions. Rates are averaged after the nonlinear
        # barrier-to-rate transformation, not by averaging A1 values first.
        nmax = self.topology["n_max"][0]
        self.table_rates = np.empty(
            (len(pressures), nmax + 1, 2), dtype=np.float64
        )
        wbarrier = params["wbarrier"]
        dbarrier = params["dbarrier"]
        vw = params["vw"]
        vd = params["vd"]
        t0 = params["t0"]

        for wet_neighbours in range(nmax + 1):
            a1_hex = self.complex_A1(wet_neighbours, 0)
            a1_oct = self.complex_A1(wet_neighbours, 8)

            omegaw_hex = np.maximum(
                wbarrier - pressures * vw
                + (nmax / 2 - wet_neighbours) * a1_hex,
                0
            )
            omegaw_oct = np.maximum(
                wbarrier - pressures * vw
                + (nmax / 2 - wet_neighbours) * a1_oct,
                0
            )
            omegad_hex = np.maximum(
                dbarrier + pressures * vd
                + (wet_neighbours - nmax / 2) * a1_hex,
                0
            )
            omegad_oct = np.maximum(
                dbarrier + pressures * vd
                + (wet_neighbours - nmax / 2) * a1_oct,
                0
            )

            fill_hex = np.exp(-omegaw_hex) / t0
            fill_oct = np.exp(-omegaw_oct) / t0
            empty_hex = np.exp(-omegad_hex) / t0
            empty_oct = np.exp(-omegad_oct) / t0

            self.table_rates[:, wet_neighbours, 0] = (
                8.0 * fill_hex + 6.0 * fill_oct
            ) / 14.0
            self.table_rates[:, wet_neighbours, 1] = (
                8.0 * empty_hex + 6.0 * empty_oct
            ) / 14.0

    def initialize_simulation(self, pressures, dt, func, params):
        # Build a compact table; the original direction-expanded table can
        # consume several GB during construction.
        self.rate_params = params.copy()
        self.tabulate_rates(func, pressures, params)
        self.dt = dt
        self.pressures = pressures
        for n in self.nodes.values():
            n.state = 0
            n.wet_neighbours = 0

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
                n_total = min(n.wet_neighbours, nmax)
                fill.append(self.table_rates[t, n_total, 0])
                empty.append(self.table_rates[t, n_total, 1])
            else:
                fill.append(0)
                empty.append(0)

        self.rates = np.array([fill, empty])

    def get_random(self):
        # Generate one timestep at a time instead of allocating a potentially
        # enormous (number of pressures x number of nodes) array.
        return np.random.rand(len(self.nodes))

    def simulate_collective(self):
        dt = self.dt
        self.average_filling = []
        for t in range(len(self.pressures)):
            states = self.states
            rates = self.rates
            random = self.get_random()
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

def get_network_stats(network):
    total_nodes = len(network.nodes)
    surface_nodes = [node.nid for node in network.get_surface()]
    nodes_with_14_neighbors = [node.nid for node in network.nodes.values() if node.n_number == 14]
    initial_wet_nodes = [node.nid for node in network.nodes.values() if node.state == 1]

    return {
        "total_nodes": total_nodes,
        "surface_nodes": surface_nodes,
        "nodes_with_14_neighbors": nodes_with_14_neighbors,
        "initial_wet_nodes": initial_wet_nodes
    }

def find_full_saturation_step(network):
    for step, filling in enumerate(network.average_filling):
        if filling == 1.0:
            return step, network.pressures[step]
    return None, None

def run_simulation(topology, params, pressures, dt, a_value_hex, a_value_oct,
                   growth_steps=5):
    seed = Node(0, [0, 0, 0], topology)
    network = Network(topology, A1_value_hex=a_value_hex, A1_value_oct=a_value_oct)
    network.add_node(seed)
    network.grow(growth_steps)

    network_stats = get_network_stats(network)

    network.initialize_simulation(pressures, dt, network.default_function, params)
    network.simulate_collective()

    saturation_step, saturation_pressure = find_full_saturation_step(network)

    return network, network_stats, saturation_step, saturation_pressure

def store_output_data(network, filename='cage_filling_data52.csv'):
    data = {
        'Time_Step': np.arange(len(network.average_filling)),
        'Pressure': network.pressures,
        'Average_Filling': network.average_filling
    }

    df = pd.DataFrame(data)
    df['Filling_Rate'] = df['Average_Filling'].diff() / df['Time_Step'].diff()
    df['Pressure_Derivative'] = df['Pressure'].diff() / df['Time_Step'].diff()

    window_size = 10
    df['MA_Filling'] = df['Average_Filling'].rolling(window=window_size).mean()
    df['MA_Filling_Rate'] = df['Filling_Rate'].rolling(window=window_size).mean()

    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def run_sensitivity_analysis(topology, params, pressures, dt, a_value_hex,
                             a_value_oct, perturbations=(-0.20, -0.10, 0.10, 0.20),
                             random_seed=42,
                             filename='sensitivity_analysis.csv'):
    """Perform a one-at-a-time sensitivity analysis of all model parameters."""
    baseline_values = {
        **params,
        "a_value_hex": a_value_hex,
        "a_value_oct": a_value_oct
    }

    def evaluate(run_params, run_a_hex, run_a_oct):
        # Common random numbers make scenario-to-scenario comparisons fair.
        np.random.seed(random_seed)
        network, _, saturation_step, saturation_pressure = run_simulation(
            topology, run_params, pressures, dt, run_a_hex, run_a_oct
        )
        hysteresis, _, _, _, _ = calculate_hysteresis(network)
        return {
            "Hysteresis": hysteresis,
            "Saturation_Step": saturation_step,
            "Saturation_Pressure": saturation_pressure,
            "Maximum_Filling": np.max(network.average_filling),
            "Final_Filling": network.average_filling[-1]
        }

    baseline_metrics = evaluate(params.copy(), a_value_hex, a_value_oct)
    rows = [{
        "Parameter": "baseline",
        "Perturbation_Percent": 0.0,
        "Baseline_Value": np.nan,
        "Perturbed_Value": np.nan,
        **baseline_metrics
    }]

    for parameter, baseline_value in baseline_values.items():
        for perturbation in perturbations:
            run_params = params.copy()
            run_a_hex, run_a_oct = a_value_hex, a_value_oct
            perturbed_value = baseline_value * (1.0 + perturbation)

            if parameter in run_params:
                run_params[parameter] = perturbed_value
            elif parameter == "a_value_hex":
                run_a_hex = perturbed_value
            else:
                run_a_oct = perturbed_value

            metrics = evaluate(run_params, run_a_hex, run_a_oct)
            rows.append({
                "Parameter": parameter,
                "Perturbation_Percent": 100.0 * perturbation,
                "Baseline_Value": baseline_value,
                "Perturbed_Value": perturbed_value,
                **metrics
            })

    results = pd.DataFrame(rows)
    for metric in ("Hysteresis", "Saturation_Pressure", "Maximum_Filling",
                   "Final_Filling"):
        baseline = baseline_metrics[metric]
        if baseline is not None and baseline != 0:
            results[f"{metric}_Change_Percent"] = (
                100.0 * (results[metric] - baseline) / baseline
            )
        else:
            results[f"{metric}_Change_Percent"] = np.nan

    results.to_csv(filename, index=False)
    print(f"Sensitivity analysis saved to {filename}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help=(
            "Run one sensitivity-analysis case. Without this switch, run "
            "the normal baseline simulation."
        )
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=0,
        help="Sensitivity case number, from 0 to 28 (requires --sensitivity)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used to derive reproducible replicate seeds"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of independent replicates to run (maximum 26)"
    )
    parser.add_argument(
        "--output-dir",
        default="results"
    )
    args = parser.parse_args()
    if not 1 <= args.n_runs <= 26:
        parser.error("--n-runs must be between 1 and 26")

    zif71 = {
        "n_max": {0: 14},
        "type_accept": {0: [0]},
        "n_directions": {0: 14},
        "directions": [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0],
            [0, 0, 2], [0, 0, -2]
        ],
        "rotation": {0: 1},
        "reverse": {
            0: 7, 1: 6, 2: 5, 3: 4,
            4: 3, 5: 2, 6: 1, 7: 0,
            8: 9, 9: 8, 10: 11, 11: 10,
            12: 13, 13: 12
        }
    }

    params = {
        "wbarrier": 44,
        "dbarrier": 2,
        "vw": 0.55,
        "vd": 0.08,
        "t0": 1
    }

    a_value_hex = 1.2
    a_value_oct = 0.87
    dt = 0.2
    growth_steps = 5

    pressures = np.concatenate([
        np.linspace(0, 100, num=120000),
        np.linspace(100, 0, num=120000)
    ])

    perturbations = (-0.20, -0.10, 0.10, 0.20)
    baseline_values = {
        **params,
        "a_value_hex": a_value_hex,
        "a_value_oct": a_value_oct
    }

    # Case 0 is the baseline. Cases 1–28 are parameter perturbations.
    cases = [("baseline", 0.0)]

    for parameter in baseline_values:
        for perturbation in perturbations:
            cases.append((parameter, perturbation))

    if args.sensitivity:
        if not 0 <= args.case_id < len(cases):
            parser.error(
                f"--case-id must be between 0 and {len(cases) - 1}"
            )
        parameter, perturbation = cases[args.case_id]
    else:
        if args.case_id != 0:
            parser.error("--case-id requires --sensitivity")
        parameter, perturbation = "baseline", 0.0

    run_params = params.copy()
    run_a_hex = a_value_hex
    run_a_oct = a_value_oct

    if parameter != "baseline":
        perturbed_value = baseline_values[parameter] * (1.0 + perturbation)

        if parameter in run_params:
            run_params[parameter] = perturbed_value
        elif parameter == "a_value_hex":
            run_a_hex = perturbed_value
        elif parameter == "a_value_oct":
            run_a_oct = perturbed_value
    else:
        perturbed_value = np.nan

    output_dir = args.output_dir
    import os
    os.makedirs(output_dir, exist_ok=True)

    case_name = (
        "baseline"
        if parameter == "baseline"
        else f"{parameter}_{perturbation:+.0%}"
    )

    # Match the SOD seed convention: base seed + growth offset + replicate
    # index. The sensitivity case is deliberately excluded so corresponding
    # replicates use common random numbers across all parameter cases.
    run_seeds = [
        args.seed + growth_steps * 10000 + run_index
        for run_index in range(args.n_runs)
    ]

    summary_rows = []
    for run_index, run_seed in enumerate(run_seeds):
        run_label = chr(ord("a") + run_index)
        np.random.seed(run_seed)
        print(
            f"Starting replicate {run_label} ({run_index + 1}/{args.n_runs}) "
            f"with seed {run_seed}"
        )

        network, network_stats, saturation_step, saturation_pressure = (
            run_simulation(
                zif71,
                run_params,
                pressures,
                dt,
                run_a_hex,
                run_a_oct,
                growth_steps=growth_steps
            )
        )
        hysteresis, _, _, _, _ = calculate_hysteresis(network)

        if args.sensitivity:
            curve_name = (
                f"filling_case_{args.case_id:02d}_{case_name}_{run_label}.csv"
            )
        else:
            curve_name = f"cage_filling_data_{run_label}.csv"
        store_output_data(network, os.path.join(output_dir, curve_name))

        summary_rows.append({
            "Mode": "sensitivity" if args.sensitivity else "normal",
            "Case_ID": args.case_id if args.sensitivity else np.nan,
            "Run": run_index + 1,
            "Run_Label": run_label,
            "Growth_Steps": growth_steps,
            "Parameter": parameter,
            "Perturbation_Percent": 100.0 * perturbation,
            "Perturbed_Value": perturbed_value,
            "Base_Seed": args.seed,
            "Random_Seed": run_seed,
            "Hysteresis": hysteresis,
            "Saturation_Step": saturation_step,
            "Saturation_Pressure": saturation_pressure,
            "Maximum_Filling": np.max(network.average_filling),
            "Final_Filling": network.average_filling[-1]
        })

    summary = pd.DataFrame(summary_rows)
    summary_name = (
        f"summary_case_{args.case_id:02d}.csv"
        if args.sensitivity else "summary_normal.csv"
    )
    summary_filename = os.path.join(output_dir, summary_name)
    summary.to_csv(summary_filename, index=False)

    print(summary.to_string(index=False))
    if args.sensitivity:
        print(
            f"Completed {args.n_runs} replicate(s) for sensitivity case "
            f"{args.case_id}: {case_name}"
        )
    else:
        print(f"Completed {args.n_runs} normal baseline replicate(s)")
