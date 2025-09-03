import sys
import os
sys.path.append(os.path.dirname(__file__))  # Add current file's directory to sys.path

from fast_scene import create_dataset
import numpy as np
import matplotlib.pyplot as plt
import cma
from utils.stats import global_stats
import time
from scipy.special import rel_entr
import json 
import time
from scipy.optimize import differential_evolution
from scipy.optimize import minimize



def load_json_config(path: str) -> dict:
    """Load JSON config from given path."""
    with open(path, 'r') as f:
        return json.load(f)
    
def stopping_condition(xk, convergence):
    # Check if the best solution is within 0.001 of the global minimum (0 in this case)
    if np.abs(convergence) < 0.001:
        print("Stopping: Reached optimal solution within tolerance.")
        return True
    return False
    
def evaluate_lum_line(params, settings):
    cfg = load_json_config(settings['base_config'])
    c = params

    cfg['lighting']['3d_scene_light_intensities']['p0'] = cfg['lighting']['3d_scene_light_intensities']['p0'] * c
    cfg['lighting']["3d_scene_light_intensities"]['p1'] = cfg['lighting']['3d_scene_light_intensities']['p1'] * c
    cfg['lighting']["3d_scene_light_intensities"]['p2'] = cfg['lighting']['3d_scene_light_intensities']['p2'] * c
    cfg['lighting']["3d_scene_light_intensities"]['p3'] = cfg['lighting']['3d_scene_light_intensities']['p3'] * c
    cfg['train_mode'] = "lum"

    res = create_dataset(cfg)

    kl_div = abs(res - settings['target_average'])
    print(f"✅ Error: {kl_div:.6f} (res = {res})")

    return kl_div

def evaluate_lum(params, config_par):
    cfg = load_json_config(config_par['base_config'])
    p0, p1, p2, p3 = params

    cfg['lighting']['3d_scene_light_intensities']['p0'] = p0
    cfg['lighting']["3d_scene_light_intensities"]['p1'] = p1
    cfg['lighting']["3d_scene_light_intensities"]['p2'] = p2
    cfg['lighting']["3d_scene_light_intensities"]['p3'] = p3
    cfg['train_mode'] = "lum"

    res = create_dataset(cfg)

    kl_div = abs(res - config_par['target_average'])
    
    print(f"✅ Error: {kl_div:.6f} (res = {res})")
    print(f"{p0}, {p1}, {p2}, {p3}")

    return kl_div

def evaluate_line(params, settings):
    config = load_json_config(settings['base_config'])
    c = params

    config['motion']['mean_translation']['x'] = config['motion']['mean_translation']['x'] * c
    config['motion']['mean_translation']['y'] = config['motion']['mean_translation']['y'] * c
    config['train_mode'] = "disp"
    config['scene']['seed'] = settings['seed']

    res = create_dataset(config)

    kl_div = abs(res - settings['target_mean'])

    return kl_div

def evaluate_disp(params, settings):
    config = load_json_config(settings['base_config'])
    x_trans, y_trans = params

    config['motion']['mean_translation']['x'] = x_trans
    config['motion']['mean_translation']['y'] = y_trans
    config['train_mode'] = "disp"
    config['scene']['seed'] = settings['seed']

    res = create_dataset(config)

    kl_div = abs(res - settings['target_mean'])
    
    print(f"Settings: {x_trans}, {y_trans}")
    print(f"✅ Error: {kl_div:.6f} (res = {res})")

    return kl_div

def dum_line(settings):
    bounds = [
        (1, 3)  # Bound for the single variable
    ]
    
    for target_lum in [0.05]:
        settings['target_average'] = target_lum

        for seed in range(49, 50):
            settings['seed'] = seed
            settings['base_config'] = f"/Users/Petter/Documents/uni/thesis/Blender2flow/cfg_lum_1_{settings['seed']}.json"

            # Define the objective function to minimize
            def objective_function(x):
                return evaluate_lum_line(x, settings)  # Assuming evaluate_line can handle x as input
            
            x0 = np.random.uniform(4, 7)

            result = minimize(objective_function, x0, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-3, 'disp': True})


            # Final result
            best_params = result.x

            cfg = load_json_config(settings['base_config'])

            cfg['lighting']['3d_scene_light_intensities']['p0'] = cfg['lighting']['3d_scene_light_intensities']['p0']* best_params[0]
            cfg['lighting']["3d_scene_light_intensities"]['p1'] = cfg['lighting']['3d_scene_light_intensities']['p1'] * best_params[0]
            cfg['lighting']["3d_scene_light_intensities"]['p2'] = cfg['lighting']['3d_scene_light_intensities']['p2'] * best_params[0]
            cfg['lighting']["3d_scene_light_intensities"]['p3'] = cfg['lighting']['3d_scene_light_intensities']['p3'] * best_params[0]

            cfg_path = f"cfg_lum_{int(settings['target_average']*100)}_{seed}.json"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
            print(f"Full configuration saved to {cfg_path}")

def train_line(settings):
    bounds = [
        (1, 10)  # Bound for the single variable
    ]
    
    for target_mean in range(30, 100, 10):
        settings['target_mean'] = target_mean

        for seed in range(50):
            settings['seed'] = seed
            settings['base_config'] = f"/Users/Petter/Documents/uni/thesis/Blender2flow/mean_10_{settings['seed']}.json"

            # Define the objective function to minimize
            def objective_function(x):
                return evaluate_line(x, settings)  # Assuming evaluate_line can handle x as input

            # Use differential evolution for optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=2000,
                popsize=15,
                tol=1e-2,  # Set tolerance on the objective function improvement
                atol=1e-2,  # Set tolerance on the solution vector (change in solutions)
            )

            # Final result
            best_params = result.x

            cfg = load_json_config(settings['base_config'])

            cfg['motion']['mean_translation']['x'] = cfg['motion']['mean_translation']['x'] * best_params[0]
            cfg['motion']['mean_translation']['y'] = cfg['motion']['mean_translation']['y'] * best_params[0]
            cfg['scene']['seed'] = seed

            cfg_path = f"mean_{target_mean}_{seed}.json"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
            print(f"Full configuration saved to {cfg_path}")


def train_disp(settings):
    bounds = [
        (0.0, 0.5),   # x_trans
        (0.0, 0.5)   # y_trans
    ]

    for seed in range(5,50):
        settings['seed'] = seed

        # Initial guess (center of the bounds)
        x0 = [np.random.uniform(bounds[0][0], bounds[0][1]), 
            np.random.uniform(bounds[1][0], bounds[1][1])]
        
        # Convert bounds to CMA format
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        
        opts = {
            'bounds': [lower_bounds, upper_bounds],
            'popsize': 5,  # higher diversity
            'CMA_active': True,
            'CMA_elitist': True,  # allow bad samples to contribute
            'CMA_mirrormethod': 1,  # reflect to stay in bounds
            'tolx': 1e-3,
            'tolfun': 1e-3,
            'maxfevals': 2000,
            'verb_disp': 1,
            'verb_log': 0
        }

        # Run optimization
        es = cma.CMAEvolutionStrategy(x0, 0.3, opts)
        while not es.stop():
            solutions = es.ask()
            values = [evaluate_disp(x, settings) for x in solutions]
            es.tell(solutions, values)
            es.logger.add()

        # Final result
        best_params = es.result.xbest

        cfg = load_json_config(settings['base_config'])

        cfg['motion']['mean_translation']['x'] = best_params[0]
        cfg['motion']['mean_translation']['y'] = best_params[1]
        cfg['scene']['seed'] = seed


        cfg_path = f"mean_{int(settings['target_mean'])}_{settings['seed']}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        print(f"Full configuration saved to {cfg_path}")

def train_lum(settings):
    bounds = [
        (0, 10), # light 1
        (0, 10), # light 2
        (0, 10), # light 3
        (0, 10) # light 4
    ]

    for seed in range(45,50):
        settings['seed'] = seed

        # Initial guess (center of the bounds)
        x0 = [np.random.uniform(bounds[0][0], bounds[0][1]), 
              np.random.uniform(bounds[1][0], bounds[1][1]),
              np.random.uniform(bounds[2][0], bounds[2][1]),
              np.random.uniform(bounds[3][0], bounds[3][1])]

        # Initial guess (center of the bounds)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
        
        # Convert bounds to CMA format
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        
        opts = {
            'bounds': [lower_bounds, upper_bounds],
            'popsize': 5,  # higher diversity
            'CMA_active': True,
            'CMA_elitist': True,  # allow bad samples to contribute
            'CMA_mirrormethod': 1,  # reflect to stay in bounds
            'tolx': 1e-3,
            'tolfun': 1e-3,
            'maxfevals': 2000,
            'verb_disp': 1,
            'verb_log': 0
        }

        # Run optimization
        es = cma.CMAEvolutionStrategy(x0, 50, opts)
        while not es.stop():
            solutions = es.ask()
            values = [evaluate_lum(x, settings) for x in solutions]
            es.tell(solutions, values)
            es.logger.add()

            # Custom stopping condition
            best_error = min(values)
            if best_error < 0.001:
                print(f"Stopping early: best error {best_error:.5f} < 0.01")
                break

        # Final result
        best_params = es.result.xbest
        cfg = load_json_config(settings['base_config'])

        cfg['lighting']['3d_scene_light_intensities']['p0'] = best_params[0]
        cfg['lighting']["3d_scene_light_intensities"]['p1'] = best_params[1]
        cfg['lighting']["3d_scene_light_intensities"]['p2'] = best_params[2]
        cfg['lighting']["3d_scene_light_intensities"]['p3'] = best_params[3]

        cfg_path = f"cfg_lum_{int(settings['target_average']*100)}_{seed}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        print(f"Full configuration saved to {cfg_path}")

def train_fluc():
    pass

def run_optimization(settings):
    mode = settings['target_indicator']
    if mode == 'displacement':
        train_disp(settings)
    elif mode == 'luminance':
        dum_line(settings)
    else:
        print("This mode does not exist!")
    
    return


if __name__ == "__main__":
    run_optimization()