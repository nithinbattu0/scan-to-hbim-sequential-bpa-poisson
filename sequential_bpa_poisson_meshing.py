import os
import sys
import time
import math
import numpy as np
import open3d as o3d

import argparse

# ---------------- ARGUMENT PARSER ----------------
parser = argparse.ArgumentParser(description="Sequential BPA–Poisson Surface Reconstruction")

parser.add_argument("--input", required=True, help="Path to input PLY file")
parser.add_argument("--output", required=True, help="Path to output mesh PLY file")
args = parser.parse_args()

INPUT_PLY = args.input
OUT_PLY = args.output
# -------------------------------------------------

# -------------------------
# Utility functions
# -------------------------
def vprint(*args, **kw):
    print(*args, **kw)
    sys.stdout.flush()

def estimate_avg_spacing(pcd, k=6):
    """Estimate median nearest-neighbor distance (robust estimate of spacing)."""
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < k+1:
        return 0.01  # tiny fallback
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(min(n, 2000)):  # sample up to 2000 points for speed
        idx = np.random.randint(0, n)
        [_, idxs, d] = tree.search_knn_vector_3d(pcd.points[idx], k+1)
        if len(d) >= 2:
            # ignore the zero distance to itself; take mean of neighbors
            nn_d = np.sqrt(d[1:]).mean()
            dists.append(nn_d)
    if len(dists) == 0:
        return 0.01
    return float(np.median(dists))

def keep_largest_component(mesh, min_tri_fraction=0.05):
    """Keep only the largest triangle-connected component."""
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) == 0:
        return mesh
    largest = int(np.argmax(cluster_n_triangles))
    triangles_to_keep = [i for i, c in enumerate(triangle_clusters) if c == largest]
    # boolean mask for triangles
    triangle_mask = np.array(cluster_n_triangles)  # not used directly; we'll create new mesh
    # create new mesh with only the largest cluster
    mesh_triangles = np.asarray(mesh.triangles)
    keep_tri_mask = np.asarray(triangle_clusters) == largest
    kept_tris = mesh_triangles[keep_tri_mask]
    new_mesh = o3d.geometry.TriangleMesh()
    # gather vertices referenced by kept_tris and reindex
    if kept_tris.shape[0] == 0:
        return mesh
    unique_verts, inverse = np.unique(kept_tris.flatten(), return_inverse=True)
    verts = np.asarray(mesh.vertices)[unique_verts]
    tris = inverse.reshape((-1, 3))
    new_mesh.vertices = o3d.utility.Vector3dVector(verts)
    new_mesh.triangles = o3d.utility.Vector3iVector(tris)
    new_mesh.compute_vertex_normals()
    return new_mesh

def crop_mesh_to_bbox(mesh, pcd, padding=0.001):
    """Crop mesh to axis-aligned bounding box of point cloud plus small padding."""
    bbox = pcd.get_axis_aligned_bounding_box()
    if padding > 0:
        minb = bbox.min_bound - padding
        maxb = bbox.max_bound + padding
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=minb, max_bound=maxb)
    try:
        mesh_crop = mesh.crop(bbox)
        return mesh_crop
    except Exception:
        return mesh

def remove_small_components(mesh, min_triangles=100):
    """Remove tiny disconnected components (by triangle count)."""
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) == 0:
        return mesh
    keep = [i for i, c in enumerate(cluster_n_triangles) if c >= min_triangles]
    if len(keep) == 0:
        # keep the largest
        largest = int(np.argmax(cluster_n_triangles))
        keep = [largest]
    mask = np.isin(triangle_clusters, keep)
    tris = np.asarray(mesh.triangles)[mask]
    unique_verts, inv = np.unique(tris.flatten(), return_inverse=True)
    verts = np.asarray(mesh.vertices)[unique_verts]
    tris = inv.reshape((-1, 3))
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(verts)
    new_mesh.triangles = o3d.utility.Vector3iVector(tris)
    new_mesh.compute_vertex_normals()
    return new_mesh

# -------------------------
# Main pipeline
# -------------------------
def main():
    t0 = time.time()
    vprint("Input:", INPUT_PLY)
    if not os.path.isfile(INPUT_PLY):
        vprint("ERROR: input file not found:", INPUT_PLY)
        return

    # Load point cloud
    vprint("[INFO] Loading point cloud...")
    pcd = o3d.io.read_point_cloud(INPUT_PLY)
    n_pts = len(pcd.points)
    if n_pts == 0:
        vprint("ERROR: no points loaded.")
        return
    vprint(f"[INFO] Points loaded: {n_pts}")

    # Estimate normals (if not present)
    vprint("[INFO] Estimating normals (auto radius)...")
    spacing = estimate_avg_spacing(pcd, k=6)
    if spacing <= 0:
        spacing = 0.01
    # radius for normals: a few times spacing
    radius_normals = spacing * 3.0
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=30))
        pcd.normalize_normals()
    except Exception as e:
        vprint("[WARN] normal estimation failed:", e)

    # Build BPA radii list automatically from spacing
    base = max(spacing * 1.5, 1e-5)
    radii = [base, base * 2.0, base * 4.0]
    vprint(f"[INFO] Estimated spacing: {spacing:.6f} -> BPA radii: {[round(r,6) for r in radii]}")

    # Attempt Ball Pivoting
    vprint("[INFO] Running Ball Pivoting Algorithm (BPA)...")
    try:
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        bpa_mesh.compute_vertex_normals()
        vprint("[INFO] BPA triangles:", len(bpa_mesh.triangles))
    except Exception as e:
        vprint("[WARN] BPA failed:", e)
        bpa_mesh = None

    # Evaluate BPA quality: count boundary edges and triangles
    def mesh_quality(mesh):
        if mesh is None:
            return {"triangles": 0, "boundary_edges": 1e9}
        tri = len(mesh.triangles)
        # boundary edges
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        # open3d doesn't provide boundary edges count directly; approximate via Euler or use vertex normals
        # use adjacency to find edges border: fallback inexpensive heuristic -> count triangles with no normals?
        # We'll approximate by number of connected components triangles to detect fragmentation
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        n_comp = len(cluster_n_triangles) if cluster_n_triangles is not None else 1
        return {"triangles": tri, "components": n_comp}

    q_bpa = mesh_quality(bpa_mesh)
    vprint(f"[INFO] BPA quality: triangles={q_bpa['triangles']}, components={q_bpa['components']}")

    final_mesh = None

    # Decide whether to accept BPA or fallback to Poisson
    accept_bpa = False
    if bpa_mesh is not None and q_bpa["triangles"] > max(3000, n_pts * 0.2) and q_bpa["components"] <= 3:
        # large triangle count and not fragmented -> accept
        accept_bpa = True

    if accept_bpa:
        vprint("[INFO] Accepting BPA result.")
        # postprocess: keep largest comp and remove tiny pieces
        m = keep_largest_component(bpa_mesh)
        m = remove_small_components(m, min_triangles=50)
        m = crop_mesh_to_bbox(m, pcd, padding=spacing * 1.0)
        final_mesh = m
    else:
        vprint("[INFO] BPA not sufficient -> running Poisson reconstruction to fill holes.")
        # Poisson depth auto: based on log2 of points, capped
        depth = int(min(max(8, math.floor(math.log2(max(n_pts, 256)) - 3)), 12))
        vprint(f"[INFO] Poisson depth chosen: {depth}")
        try:
            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            poisson_mesh.compute_vertex_normals()
            # crop to bbox
            mesh_crop = crop_mesh_to_bbox(poisson_mesh, pcd, padding=spacing * 1.0)
            # keep largest component & remove tiny pieces
            mesh_crop = keep_largest_component(mesh_crop)
            mesh_crop = remove_small_components(mesh_crop, min_triangles=max(50, int(n_pts/500)))
            final_mesh = mesh_crop
            vprint("[INFO] Poisson triangles:", len(final_mesh.triangles))
        except Exception as e:
            vprint("[ERROR] Poisson reconstruction failed:", e)
            # fallback to BPA result if available
            final_mesh = bpa_mesh if bpa_mesh is not None else None

    if final_mesh is None or len(final_mesh.triangles) == 0:
        vprint("[ERROR] No valid mesh produced.")
        return

    # final cleanups
    final_mesh.remove_unreferenced_vertices()
    final_mesh.compute_vertex_normals()

    # Save final mesh to PLY
    vprint("[INFO] Saving final mesh to:", OUT_PLY)
    o3d.io.write_triangle_mesh(OUT_PLY, final_mesh, write_ascii=False, compressed=False)
    vprint("[INFO] Saved.")


    vprint("Total time elapsed:", round(time.time() - t0,2), "s")
    vprint("[DONE] Output:", OUT_PLY)

if __name__ == "__main__":
    main()
