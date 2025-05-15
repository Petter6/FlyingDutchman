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



# === CONFIG ===
BLENDER_EXEC = "/Applications/Blender.app/Contents/MacOS/Blender"
EVAL_SCRIPT = "eval_candidate.py"
BASE_CONFIG = "./config/setting/base_config.json"


# === WRITE CONFIG ===
def generate_config():
    return {
    "num_scenes": 10,
    "num_obj": 5,
    "y_resolution": 540,
    "x_resolution": 960,
    "folder_path": "/Users/Petter/Documents/uni/thesis/datasets/lights_500",
    "backgrounds_path": "./resources/hdri",
    "objects_path": "./resources/objects/objects.blend",
    "objects_with_text": "./resources/objects/objects_with_text.blend",
    "3d_background_path": "/Users/Petter/Documents/uni/thesis/Blender2flow/resources/3d_bg/shop/Shop.blend",
    "use_3d_bg": False, 
    "tmp_exr_path": "./tmp",
    "camera_pos": [0.4609, 7.0830, 3.2841],
    "camera_rot": [1.4690, 0.0129, -9.6250],
    "camera_x_trans": 0.0,
    "camera_y_trans": 0.0,
    "camera_z_trans": 0.2,
    "camera_x_rot": 0.0,
    "camera_y_rot": 0.0,
    "camera_z_rot": 0.1,
    "light_intensity": 0,
    "light_intensity_second_frame": 0,
    "light_orientation": "left",
    "light_type": "POINT",
    "textures_enabled": True,
    "bg_path": None,
    "bg_intensity": 1.0,
    "z_min": 2.0, 
    "z_max": 3.0,
    "gaussian_trans": False,
    "sigma_trans": 0.2,
    "sigma_rot": 0.0,
    "uniform_trans": True,
    "x_trans": 0.001,
    "y_trans": 0.001,
    "z_trans": 0.001,
    "calc_occ": False,
    "calc_disp": True,
    "print_stats": True,
    "fog": False,
    "fog_perc": 0.7,
    "blur": False,
    "shutter_speed": 0.3,
    "seed": 5, 
    "train_mode": "disp",
    "target_dist": "./displacement_hist.npy",
    "light_intensity_p0": 500,
    "light_intensity_p1": 500,
    "light_intensity_p2": 500,
    "light_intensity_p3": 500,
    "inverted_colors": False,
    "inversion_strength": 0.0
    }

def compute_luminance_error(
    luminance_path,
    target_mean=50,
    verbose=True,
    light_value=None
):
    brightness = np.load(luminance_path, mmap_mode='r')  # shape: (num_scenes, H, W)

    # shape: (S, H, W)
    total_sum = brightness.sum()
    total_pixels = np.prod(brightness.shape)

    # Gemiddelde luminantie over alle scènes
    mean_luminance = total_sum / total_pixels
    error = abs(mean_luminance * 100 - target_mean)  # schaal luminantie van [0,1] naar [0,100]

    if verbose:
        print(f"✅ Mean Luminance: {mean_luminance * 100:.2f}")
        print(f"🎯 Target Luminance: {target_mean}")
        print(f"📉 Luminance Error: {error:.2f}")

        # Visualiseer eventueel
        plt.figure(figsize=(6, 3))
        plt.bar(["Mean Luminance", "Target"], [mean_luminance * 100, target_mean], color=["blue", "green"])
        plt.title(f"Luminance Error = {error:.2f} (Avg. Energy={light_value:.1f})" if light_value else f"Luminance Error = {error:.2f}")
        plt.ylim(0, 100)
        plt.ylabel("Luminance (0-100)")
        plt.tight_layout()
        plt.savefig("luminance_error_plot.png", dpi=150)
        plt.close()

    return error

def compute_kl_divergence_from_chunks(
    flow_path,
    target_path=None,
    batch_size=100,
    max_bins=124,
    target_mean=None,
    target_std=None,
    epsilon=1e-10,
    verbose=True
):
    flow = np.load(flow_path, mmap_mode='r')  # memory-mapped array
    num_scenes = flow.shape[0]
    h, w = flow.shape[1], flow.shape[2]

    hist_array = np.zeros(max_bins, dtype=np.int64)

    for start in range(0, num_scenes, batch_size):
        end = min(start + batch_size, num_scenes)
        chunk = flow[start:end]  # shape: (B, H, W, 2)

        # calculates the displacement (pythagoras)
        disp = np.linalg.norm(chunk, axis=-1).reshape(-1)
        rounded = np.round(disp).astype(np.int32)

        # only the values smaller than max_bins are used 
        valid = rounded < max_bins
        binned = np.bincount(rounded[valid], minlength=max_bins)

        binned[0] = 0.000001  # ✅ Exclude displacement = 0 (static background)
        hist_array += binned

    # Normalize
    hist_array = hist_array.astype(np.float64)
    hist_array /= hist_array.sum()

    support = np.arange(max_bins)

    # Target distribution
    if target_path:
        target_distribution = np.load(target_path)
        if target_distribution.shape[0] != max_bins:
            raise ValueError(f"Target distribution length ({target_distribution.shape[0]}) does not match max_bins ({max_bins}).")
        target_distribution = target_distribution.astype(np.float64)
    else:
        target_distribution = np.exp(-0.5 * ((support - target_mean) / target_std) ** 2)
        target_distribution /= target_distribution.sum()

    # Avoid zero entries
    hist_array += epsilon
    target_distribution += epsilon

    # Normalize again
    hist_array /= hist_array.sum()
    target_distribution /= target_distribution.sum()

    # KL divergence: D_KL(P || Q)
    kl_div = np.sum(rel_entr(hist_array, target_distribution))

    if verbose:
        print(f"✅ KL Divergence: {kl_div:.6f}")

        plt.figure(figsize=(8, 4))
        plt.plot(support, hist_array, label="Generated Histogram", linewidth=2)
        plt.plot(support, target_distribution, label="Target Distribution", linestyle="--", linewidth=2)
        plt.title(f"KL Divergence: {kl_div:.6f}")
        plt.xlabel("Displacement")
        plt.ylabel("Normalized Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("displacement_comparison_kl.png", dpi=150)
        plt.close()

    return kl_div


# === EVALUATION FUNCTION ===
def evaluate_lum(params, config):
    config = generate_config()
    p0, p1, p2, p3 = params
    config['light_intensity_p0'] = p0
    config['light_intensity_p1'] = p1
    config['light_intensity_p2'] = p2
    config['light_intensity_p3'] = p3
    config['train_mode'] = "lum"
    config['num_scenes'] = 1

    try:
        create_dataset(config)

        # Measure Wasserstein time
        start_time = time.time()
        kl_div = compute_luminance_error(
            luminance_path=",/tmp/brightness_maps.npy",
            target_mean=config['target_average'],
            light_value=(p0+p1+p2+p3/4)
        )
        elapsed = time.time() - start_time
        print(f"✅ KL Divergence: {kl_div:.6f} (computed in {elapsed:.2f} seconds)")

        return kl_div

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1e6


def evaluate_disp(params, config_par):
    config = generate_config()
    camera_x_trans, camera_y_trans, camera_z_trans, camera_x_rot, camera_y_rot, camera_z_rot, x_trans, y_trans, z_trans, num_obj, z_min, z_max = params

    config['camera_x_trans'] = camera_x_trans
    config['camera_y_trans'] = camera_y_trans
    config['camera_z_trans'] = camera_z_trans
    config['camera_x_rot'] = camera_x_rot
    config['camera_y_rot'] = camera_y_rot
    config['camera_z_rot'] = camera_z_rot
    config['x_trans'] = x_trans
    config['y_trans'] = y_trans
    config['z_trans'] = z_trans
    config['z_min'] = z_min
    config['z_max'] = z_max
    config['num_obj'] = int(round(num_obj))
    config['train_mode'] = "disp"

   
    create_dataset(config)

    # Measure Wasserstein time
    start_time = time.time()
    kl_div = compute_kl_divergence_from_chunks(
        flow_path="./tmp/all_flows.npy",
        target_path=None,
        batch_size=100,
        max_bins=124,
        target_mean=config_par['target_mean'],
        target_std=config_par['target_variation']
    )
    elapsed = time.time() - start_time
    print(f"✅ KL Divergence: {kl_div:.6f} (computed in {elapsed:.2f} seconds)")

    return kl_div

  

def train_disp(config):
    bounds = [
        (0.0, 0.2),   # x_trans
        (0.0, 0.2),   # y_trans
        (0.0, 0.2),   # z_trans
        (0.0, 0.2),       # x_rot
        (0.0, 0.2),    # y_rot
        (0.0, 0.2),      # z_rot
        (0.0, 0.5),   # x_trans obj
        (0.0, 0.5),   # y_trans obj
        (0.0, 0.5),     # z_trans
        (0, 25),    # num_ib
        (0.0, 2), #z_min
        (0.0, 2) #z_max
    ]

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
    es = cma.CMAEvolutionStrategy(x0, 0.3, opts)
    while not es.stop():
        solutions = es.ask()
        values = [evaluate_disp(x, config) for x in solutions]
        es.tell(solutions, values)
        es.logger.add()

    # Final result
    best_params = es.result.xbest

    print(best_params)

    # Zet best_params om in een JSON-vriendelijke structuur
    param_dict = {f"param_{i}": val for i, val in enumerate(best_params)}

    # Pad naar output-bestand
    output_path = os.path.join("best_params.json")

    # Schrijf naar JSON
    with open(output_path, "w") as f:
        json.dump(param_dict, f, indent=4)

    print(f"Best parameters saved to {output_path}")

def train_lum(config):
    bounds = [
        (0, 200), # light 1
        (0, 200), # light 2
        (0, 200), # light 3
        (0, 200) # light 4
    ]

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
        values = [evaluate_lum(x, config) for x in solutions]
        es.tell(solutions, values)
        es.logger.add()

        # Custom stopping condition
        best_error = min(values)
        if best_error < 0.01:
            print(f"Stopping early: best error {best_error:.5f} < 0.01")
            break

    # Final result
    best_params = es.result.xbest
    best_config = generate_config(best_params)
    with open(config['output_file'], "w") as f:
        json.dump(best_config, f, indent=4)

    print("\n🎯 CMA-ES Best Config:")
    for k, v in best_config.items():
        print(f"{k}: {v}")

def train_fluc():
    pass

def run_optimization(config):
    mode = config['target_indicator']
    if mode == 'displacement':
        train_disp(config)
    elif mode == 'luminance':
        train_lum(config)
    else:
        print("This mode does not exist!")
    
    return


if __name__ == "__main__":
    run_optimization()