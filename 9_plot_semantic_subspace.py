"""
Script 8: 3D Visualization of Sense Trajectories Over Time (PCA + UMAP)
========================================================================

Creates 3D visualizations:
- X axis: PCA1 / UMAP1
- Y axis: TIME (timespan)
- Z axis: PCA2 / UMAP2

Plots per lexeme:
1) All projected samples
2) Smoothed centroid trajectories with confidence ellipses

Usage:
    python 8_plot_semantic_subspace.py \
        --projections-dir outputs/projections \
        --output outputs/trajectory_plots \
        [--test-first]
"""

import argparse
import logging
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import umap
from scipy.stats import gaussian_kde
from skimage import measure


import warnings
warnings.filterwarnings("ignore", message="Tight layout not applied.*")


# -----------------------
# Logging
# -----------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------
# Helpers
# -----------------------
def build_sense_color_map(all_senses):
    palette = sns.color_palette("tab10", n_colors=len(all_senses))
    return {sense: palette[i] for i, sense in enumerate(sorted(all_senses))}


def build_timespan_axis(timespans, scale=3.0):
    return {ts: i * scale for i, ts in enumerate(sorted(timespans))}


def smooth_trajectory(points, window=3, poly=1):
    if len(points) < window:
        return points
    return np.vstack([savgol_filter(points[:, d], window_length=window, polyorder=poly)
                      for d in range(points.shape[1])]).T


def compute_centroid(points):
    return np.mean(points, axis=0)


def confidence_ellipse_2d(points, n_std=2.0):
    if len(points) < 3:
        return None
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    return width, height, angle


def plot_confidence_ellipse_3d(ax, mean, cov, z, color, n_std=0.5, n_points=100, alpha=0.6):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.stack([n_std*np.sqrt(eigvals[0])*np.cos(theta),
                        n_std*np.sqrt(eigvals[1])*np.sin(theta)])
    ellipse = eigvecs @ ellipse
    ellipse[0] += mean[0]
    ellipse[1] += mean[1]
    z_vals = np.full(n_points, z)
    ax.plot(ellipse[0], np.full_like(ellipse[0], z), ellipse[1], color=color, alpha=alpha, linewidth=1.5)

def plot_density_contour_3d(
    ax,
    points_2d,
    y,
    color,
    level=0.4,
    grid_size=200,
    alpha=1.0,
    linewidth=1
):
    """
    Draw a single KDE contour as a 3D curve
    (x = dim1, y = time, z = dim2)
    """
    if len(points_2d) < 5:
        return  # too few points for density

    x = points_2d[:, 0]
    z = points_2d[:, 1]

    kde = gaussian_kde(np.vstack([x, z]))

    xmin, xmax = x.min(), x.max()
    zmin, zmax = z.min(), z.max()

    X, Z = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(zmin, zmax, grid_size)
    )

    positions = np.vstack([X.ravel(), Z.ravel()])
    density = kde(positions).reshape(X.shape)

    # normalize for stable thresholding
    density /= density.max()

    contours = measure.find_contours(density, level=level)

    for contour in contours:
        cx = X[contour[:, 0].astype(int), contour[:, 1].astype(int)]
        cz = Z[contour[:, 0].astype(int), contour[:, 1].astype(int)]
        cy = np.full_like(cx, y)

        ax.plot(cx, cy, cz, color=color, alpha=alpha, linewidth=linewidth)


# -----------------------
# Core plotting
# -----------------------
def plot_all_samples_3d(lexeme, data, embedding, sense_colors, time_pos, output_dir, suffix="pca"):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    for ts in sorted(data["timespan_data"].keys()):
        y = time_pos[ts]
        ts_data = data["timespan_data"][ts]
        if ts_data["projections"].shape[0] == 0:
            continue
        points_2d = embedding.transform(ts_data["projections"])
        points_2d -= points_2d.mean(axis=0)  # center
        for point, sense in zip(points_2d, ts_data["senses"]):
            ax.scatter(point[0], y, point[1], color=sense_colors[sense], alpha=0.35, s=20)
    ax.set_box_aspect([1, 3, 1])
    ax.view_init(elev=30, azim=-60)
    ax.set_title(f"{lexeme}: All Samples ({suffix.upper()})", fontsize=14)
    ax.set_xlabel(f"{suffix.upper()} 1")
    ax.set_ylabel("Time")
    ax.set_zlabel(f"{suffix.upper()} 2")
    ax.set_yticks(list(time_pos.values()))
    ax.set_yticklabels(time_pos.keys())
    plt.tight_layout()
    path = output_dir / f"{lexeme}_all_samples_{suffix}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info(f"  ✓ Saved {path}")


def plot_centroid_trajectories_3d(lexeme, data, timespans, embedding, sense_colors, time_pos, output_dir, suffix="pca"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    sense_points = {}
    #for ts in sorted(data["timespan_data"].keys()):
    for ts in timespans:
        y = time_pos[ts]
        ts_data = data["timespan_data"][ts]
        if ts_data["projections"].shape[0] == 0:
            continue
        points_2d = embedding.transform(ts_data["projections"])
        points_2d -= points_2d.mean(axis=0)
        for sense in set(ts_data["senses"]):
            idx = [i for i, s in enumerate(ts_data["senses"]) if s == sense]
            pts = points_2d[idx]
            if len(pts) == 0:
                continue
            centroid = compute_centroid(pts)
            sense_points.setdefault(sense, []).append((y, centroid, pts))
            
            
            # contour
            #plot_density_contour_3d(
            #    ax=ax,
            #    points_2d=points_2d,
            #    y=y,
            #    color=sense_colors[sense],
            #    level=0.6
            #)
            
            # ellipse
            cov_matrix = np.cov(pts.T)
            if len(pts) > 1:
                cov_matrix = np.cov(pts.T)
                if np.all(np.isfinite(cov_matrix)):
                        plot_confidence_ellipse_3d(ax=ax, mean=centroid[:2], cov=cov_matrix, z=y,
                                                   color=sense_colors[sense], n_std=0.5)
            
    for sense, entries in sense_points.items():
        entries = sorted(entries, key=lambda x: x[0])
        y_vals = np.array([e[0] for e in entries])
        centroids = np.array([e[1] for e in entries])
        centroids_smooth = smooth_trajectory(centroids)
        ax.plot(centroids_smooth[:, 0], y_vals, centroids_smooth[:, 1],
                color=sense_colors[sense], linewidth=3, label=sense)
    ax.set_box_aspect([1, 3, 1])
    ax.view_init(elev=40, azim=-40)
    ax.set_title(f"{lexeme}: Sense Trajectories ({suffix.upper()})", fontsize=14)
    ax.set_xlabel(f"{suffix.upper()} 1")
    ax.set_ylabel("")
    ax.set_zlabel(f"{suffix.upper()} 2")
    ax.set_yticks(list(time_pos.values()))
    ax.set_yticklabels(time_pos.keys(), ha='left')
    ax.legend(loc="best")
    plt.tight_layout()
    path = output_dir / f"{lexeme}_centroid_trajectories_{suffix}.pdf"
    plt.savefig(path, dpi=400)
    plt.close()
    logging.info(f"  ✓ Saved {path}")


# -----------------------
# Main processing
# -----------------------
def process_file(file_path: Path, output_dir: Path, exclude):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    lexeme = data["lexeme"]
    logging.info(f"Visualizing {lexeme}")
    
    # Filter out the specific unwanted strings
    #exclude = {"1995-2014", "2006-2023"}
    exclude_set = set(exclude)

    timespans = sorted([
        ts for ts in data["timespan_data"].keys() 
        if ts not in exclude_set
    ])
    #timespans = sorted(data["timespan_data"].keys())
    
    time_pos = build_timespan_axis(timespans, scale=3.0)
    all_proj = []
    all_senses = []
    for ts in timespans:
        all_proj.extend(data["timespan_data"][ts]["projections"])
        all_senses.extend(data["timespan_data"][ts]["senses"])
    all_proj = np.array(all_proj)
    if len(all_proj) < 5:
        logging.warning(f"  Not enough data for {lexeme}, skipping")
        return
    sense_colors = build_sense_color_map(set(all_senses))

    # ---------------- PCA
    pca = PCA(n_components=2)
    pca.fit(all_proj)
    #plot_all_samples_3d(lexeme, data, pca, sense_colors, time_pos, output_dir, suffix="pca")
    plot_centroid_trajectories_3d(lexeme, data, timespans, pca, sense_colors, time_pos, output_dir, suffix="pca")

    # ---------------- UMAP
    #umap_model = umap.UMAP(n_components=2)
    #umap_2d = umap_model.fit_transform(all_proj)
    #class UMAPWrapper:
    #    def __init__(self, embedding):
    #        self.embedding = embedding
    #    def transform(self, X):
    #        return self.embedding
    #umap_wrapper = UMAPWrapper(umap_2d)
    #plot_all_samples_3d(lexeme, data, umap_wrapper, sense_colors, time_pos, output_dir, suffix="umap")
    #plot_centroid_trajectories_3d(lexeme, data, umap_wrapper, sense_colors, time_pos, output_dir, suffix="umap")


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-first", action="store_true",
                        help="Only plot first projection file for testing")
    parser.add_argument("--exclude", nargs='+', default=[], help="List of timespans to ignore")
    args = parser.parse_args()

    setup_logging()
    sns.set_style("whitegrid")
    args.output.mkdir(parents=True, exist_ok=True)

    files = list(args.projections_dir.glob("*_projections_by_time.pkl"))
    if args.test_first:
        files = files[:1]

    logging.info(f"Found {len(files)} projection files")
    for f in files:
        process_file(f, args.output, args.exclude)


if __name__ == "__main__":
    main()
