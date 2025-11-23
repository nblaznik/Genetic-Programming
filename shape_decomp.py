import os
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import time

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------

# SOURCE = "uni_school_shapes.png"
# SOURCE = "nejc_shapes_nobg.png"
SOURCE = "skills_shapes.png"

NUM_CANDIDATES = 10        # mutants per generation
MAX_POLYGONS = 200         # upper limit
INITIAL_POLYGONS = 200     # starting polygon count
ITERATIONS = 50000         # how long to run
SAVE_INTERVAL = 200        # save snapshot every N steps

SAVE_DIR = f"images_{SOURCE[:-4]}_{INITIAL_POLYGONS}_{ITERATIONS}"
os.makedirs(SAVE_DIR, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


import json

# For saving the coordinates of the triangles 
def save_dna(dna, path="dna.json"):
    serializable = []
    for poly in dna:
        serializable.append({
            "points": [list(p) for p in poly["points"]],
            "color": list(poly["color"])
        })
    with open(path, "w") as f:
        json.dump(serializable, f)


# ----------------------------------------------------------
# Rendering (CPU, Pillow)
# ----------------------------------------------------------

def render_dna(dna, size):
    """Render polygons in dna onto a RGB image (Pillow)."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    for poly in dna:
        draw.polygon(poly["points"], fill=poly["color"])

    return img


# ----------------------------------------------------------
# Fitness (GPU)
# ----------------------------------------------------------

def compute_edge_weight(target_np):
    gx = cv2.Sobel(target_np, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(target_np, cv2.CV_32F, 0, 1)
    edge = np.abs(gx) + np.abs(gy)
    w = 1 + 0.5 * np.mean(edge, axis=2) / (np.max(edge) + 1e-6)
    return w

def fitness_torch(target_t, canvas_t, edge_weight_t):
    """
    target_t, canvas_t: [H, W, 3] float32 tensors on device
    edge_weight_t: [H, W] float32 tensor on device
    """
    diff = torch.mean((target_t - canvas_t) ** 2, dim=2)
    weighted = diff * edge_weight_t
    return weighted.mean()


# ----------------------------------------------------------
# DNA: polygons
# ----------------------------------------------------------

def random_polygon(W, H):
    n = 3 #random.choice([3, 4, 5, 6])
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)

    pts = []
    for _ in range(n):
        ang = random.uniform(0, 2 * np.pi)
        rad = random.uniform(5, 60)
        x = int(cx + rad * np.cos(ang))
        y = int(cy + rad * np.sin(ang))
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        pts.append((x, y))

    # color = (
    #     random.randint(0, 255),
    #     random.randint(0, 255),
    #     random.randint(0, 255),
    #     random.randint(30, 160)
    # )

    color = (
        105, 67, 21, random.randint(50, 255)
    )

    return {"points": pts, "color": color}


def mutate_polygon(poly, W, H):
    new = {
        "points": [p for p in poly["points"]],
        "color": poly["color"]
    }

    # move one point
    if random.random() < 0.7:
        idx = random.randrange(len(new["points"]))
        x, y = new["points"][idx]
        x += random.randint(-10, 10)
        y += random.randint(-10, 10)
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        new["points"][idx] = (x, y)

    # mutate color
    if random.random() < 0.3:
        r, g, b, a = new["color"]
        r = max(0, min(255, r + random.randint(-20, 20)))
        g = max(0, min(255, g + random.randint(-20, 20)))
        b = max(0, min(255, b + random.randint(-20, 20)))
        a = max(0, min(255, a + random.randint(-10, 10)))
        new["color"] = (r, g, b, a)

    return new


def mutate_dna(dna, W, H):
    new = [p.copy() for p in dna]

    # deletion
    if len(new) > 5 and random.random() < 0.05:
        idx = random.randrange(len(new))
        new.pop(idx)

    # addition
    if len(new) < MAX_POLYGONS and random.random() < 0.10:
        new.append(random_polygon(W, H))

    # mutate one polygon
    idx = random.randrange(len(new))
    new[idx] = mutate_polygon(new[idx], W, H)

    return new


# ----------------------------------------------------------
# Candidate evaluation helper (CPU render + GPU fitness)
# ----------------------------------------------------------

def evaluate_dna(dna, size, target_t, edge_weight_t):
    """
    Render dna, convert to tensor on device, compute fitness.
    Returns (fitness_value_float, PIL_image, tensor_on_device)
    """
    img = render_dna(dna, size)
    img_np = np.asarray(img).astype(np.float32)
    canvas_t = torch.from_numpy(img_np).to(device)
    f = fitness_torch(target_t, canvas_t, edge_weight_t)
    return f.item(), img, canvas_t


# ----------------------------------------------------------
# Evolution loop
# ----------------------------------------------------------

def evolve():
    # Also time it: 

    start_time = time.time()

    # load target
    target = Image.open(SOURCE).convert("RGB")
    W, H = target.size
    size = (W, H)

    # target as numpy and tensor
    target_np = np.asarray(target).astype(np.float32)
    target_t = torch.from_numpy(target_np).to(device)

    # edge weights
    edge_weight_np = compute_edge_weight(target_np)
    edge_weight_t = torch.from_numpy(edge_weight_np.astype(np.float32)).to(device)

    # initial DNA
    dna = [random_polygon(W, H) for _ in range(INITIAL_POLYGONS)]
    best_fit, best_img, best_t = evaluate_dna(dna, size, target_t, edge_weight_t)

    print("Initial fitness:", best_fit, "polygons:", len(dna))
    all_best_fits = []

    ## Probably best to evaluate based on the difference in the quality of the fit (to do)
    for step in range(ITERATIONS):
        # propose mutants
        best_local_fit = float("inf")   
        best_local_dna = None
        best_local_img = None
        best_local_t = None

        for _ in range(NUM_CANDIDATES):
            mutant = mutate_dna(dna, W, H)
            f_val, img, img_t = evaluate_dna(mutant, size, target_t, edge_weight_t)

            if f_val < best_local_fit:
                best_local_fit = f_val
                best_local_dna = mutant
                best_local_img = img
                best_local_t = img_t

        # accept if better than global best
        if best_local_fit < best_fit:
            dna = best_local_dna
            best_img = best_local_img
            best_t = best_local_t
            best_fit = best_local_fit

        print(
            f"Step {step}: improved -> {best_fit:.3f}, "
            f"polygons={len(dna)}"
        )
        all_best_fits.append(best_fit)

        if step % SAVE_INTERVAL == 0:
            best_img.save(os.path.join(SAVE_DIR, f"step_{step:06d}.png"))

    # best_img.save(os.path.join(SAVE_DIR, "final.png"))
    save_dna(dna, f"{SAVE_DIR}/final_dna.json")
    best_img.save(f"{SAVE_DIR}/final.png")
    np.save(f"{SAVE_DIR}/all_best_fits.npy", np.array(all_best_fits))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time: {elapsed/60:.2f} minutes")
    print("Done. Final fitness:", best_fit, "polygons:", len(dna))


if __name__ == "__main__":
    evolve()
