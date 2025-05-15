import numpy as np
from collections import Counter
from utils.stats import global_stats

def calculate_displacement(height, width, flow):
    # Compute displacement magnitude per pixel
    displacement = np.linalg.norm(flow, axis=2)
    # Round to nearest integer
    rounded_disp = np.round(displacement).astype(int)
    # Flatten and count using Counter
    disp_count = Counter(rounded_disp.flatten())

    global_stats.update("disp_hist", disp_count)

    
   