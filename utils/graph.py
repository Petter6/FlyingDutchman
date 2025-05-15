import numpy as np
import subprocess
from scipy.optimize import differential_evolution
from scipy.stats import norm
import matplotlib.pyplot as plt
from scene.scene_generator import create_dataset
import importlib
import config

CONFIG_PATH = 'config.py'  # must be imported by generate_scene.py
HISTOGRAM_PATH = 'displacement_hist.npy'  # output from your rendering pipeline
TARGET_MEAN = 50
TARGET_STD = 1


# 3. Load histogram
hist = np.load(HISTOGRAM_PATH)
hist = hist[:124]  # clip to consistent range
hist = hist.astype(np.float64)
hist /= hist.sum()
emp_cdf = np.cumsum(hist)

# 4. Create target CDF
x_bins = np.arange(len(emp_cdf))
target_cdf = norm.cdf(x_bins, loc=TARGET_MEAN, scale=TARGET_STD)

# 5. Compute KS distance
ks = np.max(np.abs(emp_cdf - target_cdf))

print(ks)

# (Optional) Plot for debugging
plt.plot(x_bins, emp_cdf, label="Empirical")
plt.plot(x_bins, target_cdf, label=f"N({TARGET_MEAN}, {TARGET_STD})")
plt.legend()
plt.title(f"KS: {ks:.4f}")
plt.savefig("cdf_debug.png")
plt.show()
