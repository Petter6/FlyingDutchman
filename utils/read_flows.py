import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Helper: Read .flo file (Middlebury format) ===
def read_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {filename}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        return np.reshape(data, (h, w, 2))  # (H, W, 2)

# === Collect all .flo file paths ===
def get_all_flo_paths(root_dir):
    flo_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.flo'):
                flo_paths.append(os.path.join(dirpath, file))
    return sorted(flo_paths)

# === Process .flo files and accumulate displacement magnitudes ===
def compute_displacement_histogram(flo_files, max_disp=124, bins=124, save_path="sintel_histogram.npy"):
    all_displacements = []

    for path in tqdm(flo_files, desc="Processing .flo files"):
        flow = read_flo_file(path)
        displacement = np.linalg.norm(flow, axis=2)  # shape (H, W)
        all_displacements.extend(displacement.flatten())

    all_displacements = np.array(all_displacements)
    all_displacements = all_displacements[all_displacements < max_disp]

    # Histogram
    hist, bin_edges = np.histogram(
        all_displacements,
        bins=bins,
        range=(0, max_disp),
        density=False
    )

    # Normalize
    hist = hist.astype(np.float64)
    hist /= hist.sum()

    # Save as .npy
    np.save(save_path, hist)
    print(f"✅ Histogram saved to: {save_path}")

    # Plot
    plt.figure(figsize=(8, 4))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, hist, label="Sintel Displacement Histogram")
    plt.title("Displacement Magnitude Distribution (Sintel)")
    plt.xlabel("Displacement")
    plt.ylabel("Normalized Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sintel_displacement_distribution.png", dpi=150)
    plt.show()

    return hist, bin_edges

# === Main ===
if __name__ == "__main__":
    sintel_flow_dir = "/Users/Petter/Downloads/MPI-Sintel-complete (1)/training/flow"  # Update if needed
    flo_files = get_all_flo_paths(sintel_flow_dir)
    compute_displacement_histogram(flo_files, save_path="sintel_histogram.npy")