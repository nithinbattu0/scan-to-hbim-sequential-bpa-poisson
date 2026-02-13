import os, time, math, random
import numpy as np
import open3d as o3d

import argparse

# ---------------- ARGUMENT PARSER ----------------
parser = argparse.ArgumentParser(description="Point Cloud Preprocessing for Scan-to-HBIM")

parser.add_argument("--input", required=True, help="Path to input PLY file")
parser.add_argument("--output", required=True, help="Path to output PLY file")

args = parser.parse_args()

INPUT_PATH = args.input
OUT_PATH = args.output
# ------------------------------------------------

# Visual / density tuning (change these for lighter/darker)
POINT_SIZE = 1.2          # on-screen point radius (smaller -> lighter look). Try 0.8 - 2.5
VOXEL_SCALE = 1.1        # voxel = avg_spacing * VOXEL_SCALE ; larger => fewer points
N_SAMPLE_FACTOR = 0.8    # sampled points from Poisson = len(downsampled) * N_SAMPLE_FACTOR
POISSON_DEPTH = 8        # keep 7-9 for medium fidelity; 10+ for high detail (heavy)
# ------------------------------------------------

def vprint(*a, **k):
    print(*a, **k)

def estimate_local_spacing(pcd, sample_count=800):
    n = len(pcd.points)
    if n == 0: return 1e-3
    m = min(sample_count, n)
    pts = np.asarray(pcd.points)
    idx = np.random.choice(n, m, replace=False)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dlist = []
    for i in idx:
        try:
            _, _, d2 = kdt.search_knn_vector_3d(pts[i], 7)
        except Exception:
            continue
        if len(d2) >= 2:
            dlist.append(np.mean(np.sqrt(np.array(d2[1:]))))
    return float(np.mean(dlist)) if dlist else 1e-3

def sample_feature_keypoints(pcd, sample_limit=8000, keep_min=400):
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if n == 0: return np.array([], dtype=int)
    s = min(sample_limit, n)
    sampled = np.random.choice(n, s, replace=False)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    curv = np.zeros(s, dtype=float)
    for i, idx in enumerate(sampled):
        neigh_k = min(64, max(12, int(round((n ** (1/3.0)) * 2))))
        try:
            _, neigh_idx, _ = kdt.search_knn_vector_3d(pts[idx], neigh_k)
        except Exception:
            curv[i] = 0.0
            continue
        if len(neigh_idx) < 3:
            curv[i] = 0.0
            continue
        neigh = pts[neigh_idx]
        cov = np.cov(neigh.T)
        w, _ = np.linalg.eigh(cov)
        w = np.clip(w, 0, None)
        ssum = w.sum()
        curv[i] = (w[0] / ssum) if ssum > 0 else 0.0
    keep_num = max(keep_min, int(s * 0.05))
    top = np.argsort(curv)[-keep_num:]
    return np.unique(sampled[top])

def poisson_reconstruct_and_sample(pcd, depth=8, n_samples=None, crop_to_bbox=True):
    if n_samples is None:
        n_samples = len(pcd.points)
    try:
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(20)
    except Exception:
        pass
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    except Exception as e:
        vprint("Poisson build failed:", e)
        return np.zeros((0,3), dtype=float)
    if crop_to_bbox:
        try:
            bbox = pcd.get_axis_aligned_bounding_box()
            minb = bbox.min_bound - bbox.get_extent() * 0.02
            maxb = bbox.max_bound + bbox.get_extent() * 0.02
            verts = np.asarray(mesh.vertices)
            inside = np.all((verts >= minb) & (verts <= maxb), axis=1)
            tris = np.asarray(mesh.triangles)
            tri_mask = inside[tris].any(axis=1)
            remove_mask = np.logical_not(tri_mask).astype(np.bool_)
            if remove_mask.any():
                mesh.remove_triangles_by_mask(remove_mask.tolist())
                mesh.remove_unreferenced_vertices()
        except Exception as e:
            vprint("Mesh crop warning:", e)
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        vprint("Poisson mesh empty after crop.")
        return np.zeros((0,3), dtype=float)
    try:
        sampled_pcd = mesh.sample_points_uniformly(number_of_points=int(n_samples))
        return np.asarray(sampled_pcd.points)
    except Exception as e:
        vprint("Mesh sampling failed:", e)
        return np.zeros((0,3), dtype=float)

def run_pipeline(input_path, out_path):
    t0 = time.time()
    vprint("=== START ===")
    if not os.path.exists(input_path):
        vprint("Input not found:", input_path); return
    pcd = o3d.io.read_point_cloud(input_path)
    n_orig = len(pcd.points)
    vprint("Original points:", n_orig)
    if n_orig == 0: vprint("Empty cloud."); return

    avg_spacing = estimate_local_spacing(pcd, sample_count=800)
    vprint("Estimated spacing:", f"{avg_spacing:.6e}")

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent()))
    voxel = max(avg_spacing * VOXEL_SCALE, 1e-6)
    voxel = min(voxel, diag * 0.2)
    vprint("Chosen voxel:", f"{voxel:.6e}")

    vprint("Sampling feature keypoints...")
    feat_inds = sample_feature_keypoints(pcd, sample_limit=8000, keep_min=500)
    vprint("Feature keypoints:", feat_inds.shape[0])
    keypoints = None
    if feat_inds.size:
        kp_pts = np.asarray(pcd.points)[feat_inds]
        keypoints = o3d.geometry.PointCloud(); keypoints.points = o3d.utility.Vector3dVector(kp_pts)

    vprint("Downsampling core...")
    pcd_down = pcd.voxel_down_sample(voxel)
    vprint("After downsample:", len(pcd_down.points))

    try:
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=60))
        pcd_down.orient_normals_consistent_tangent_plane(20)
    except Exception:
        pass

    vprint("Poisson reconstruct & sampling...")
    n_sample_target = max(int(len(pcd_down.points) * N_SAMPLE_FACTOR), 20000)
    sampled_pts = poisson_reconstruct_and_sample(pcd_down, depth=POISSON_DEPTH, n_samples=n_sample_target, crop_to_bbox=True)
    if sampled_pts.size == 0:
        vprint("Poisson fallback to downsampled cloud.")
        pcd_filled = pcd_down
    else:
        pcd_filled = o3d.geometry.PointCloud(); pcd_filled.points = o3d.utility.Vector3dVector(sampled_pts)
        vprint("Sampled from mesh:", len(pcd_filled.points))

    if keypoints is not None and len(keypoints.points) > 0:
        vprint("Merging keypoints and dedup...")
        merged_pts = np.vstack((np.asarray(pcd_filled.points), np.asarray(keypoints.points)))
        merged = o3d.geometry.PointCloud(); merged.points = o3d.utility.Vector3dVector(merged_pts)
        dedup_voxel = max(voxel * 0.25, avg_spacing * 0.12)
        merged = merged.voxel_down_sample(dedup_voxel)
        pcd_final = merged
    else:
        pcd_final = pcd_filled

    vprint("After merge:", len(pcd_final.points))

    try:
        nb = max(8, int(round((len(pcd_final.points) ** (1/3.0)) * 1.5)))
        pcd_final, ind = pcd_final.remove_statistical_outlier(nb_neighbors=nb, std_ratio=2.0)
        vprint("After SOR:", len(pcd_final.points), f"(nb={nb})")
    except Exception as e:
        vprint("SOR failed:", e)

    try:
        pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=80))
        pcd_final.orient_normals_consistent_tangent_plane(20)
    except Exception:
        pass

    final_n = len(pcd_final.points)
    pcd_final.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 0.0], (final_n, 1)))

    try:
        o3d.io.write_point_cloud(out_path, pcd_final)
        vprint("Saved:", out_path)
    except Exception as e:
        vprint("Save failed:", e)

    vprint("\nCounts: orig:", n_orig, "down:", len(pcd_down.points),
           "poisson:", len(pcd_filled.points), "final:", len(pcd_final.points), "\n")

    vprint("Opening viewer (black bg, strong white pts)...")
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Open 3D", width=1400, height=900)
        vis.add_geometry(pcd_final)

        # Force white color properly
        pcd_final.paint_uniform_color([1, 1, 1])

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        opt.point_size = 1.0       
        opt.light_on = False        

        # Auto-fit camera properly
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)

        vis.update_renderer()
        vis.run()
        vis.destroy_window()

    except Exception as e:
        vprint("Viewer failed (OpenGL). File saved. Error:", e)



    vprint("Total time:", round(time.time()-t0,2), "s")

if __name__ == "__main__":
    run_pipeline(INPUT_PATH, OUT_PATH)
