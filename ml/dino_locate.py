"""
Zero-training placement: DINOv2 features of box-picture cells vs. real piece patches.
Usage: python dino_locate.py <patch_dir>
"""
import csv, math, os, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F

pd = sys.argv[1]
CELL, WIN, RES = 64, 128, 224
torch.set_num_threads(16)
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
core = np.zeros((WIN, WIN), np.uint8); core[26:102, 26:102] = 255  # corner square + a margin (inner tab/blank area)

def prep(bgr, mask):
    # Central square only; holes (blanks, outline errors) inpainted from surrounding piece pixels.
    img = bgr[30:98, 30:98].copy(); mk = mask[30:98, 30:98]
    if (mk < 128).any() and (mk >= 128).any():
        img = cv2.inpaint(img, (mk < 128).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)

@torch.no_grad()
def embed(batch):
    x = (torch.stack(batch) - MEAN) / STD
    out = m.forward_features(x)
    cls = F.normalize(out["x_norm_clstoken"], dim=1)
    tok = out["x_norm_patchtokens"].view(len(batch), 16, 16, -1).permute(0, 3, 1, 2)
    grid = F.adaptive_avg_pool2d(tok, 4).flatten(2)          # (B, D, 16)
    grid = F.normalize(grid, dim=1).flatten(1) / 4             # unit-norm overall
    return cls, grid

poster = cv2.imread(os.path.join(pd, "poster64.png"))
PH, PW = poster.shape[:2]; COLS = PW // CELL; ROWS = math.ceil(PH / CELL)
pad = cv2.copyMakeBorder(poster, WIN, WIN, WIN, WIN, cv2.BORDER_REFLECT)
t0 = time.time()
cells_cls, cells_grid, cell_ids = [], [], []
batch, ids = [], []
for r in range(ROWS):
    for c in range(COLS):
        cx, cy = (c + 0.5) * CELL, (r + 0.5) * CELL
        x0, y0 = int(cx - WIN / 2) + WIN, int(cy - WIN / 2) + WIN
        win = pad[y0:y0 + WIN, x0:x0 + WIN]
        for k in range(4):  # patch = poster rotated k quarter turns clockwise
            w = win; mk = core
            for _ in range(k):
                w = cv2.rotate(w, cv2.ROTATE_90_CLOCKWISE)
            batch.append(prep(w, mk)); ids.append((r * COLS + c) * 4 + k)
            if len(batch) == 64:
                a, b = embed(batch); cells_cls.append(a); cells_grid.append(b); cell_ids += ids; batch, ids = [], []
if batch:
    a, b = embed(batch); cells_cls.append(a); cells_grid.append(b); cell_ids += ids
cells_cls = torch.cat(cells_cls); cells_grid = torch.cat(cells_grid); cell_ids = torch.tensor(cell_ids)
print(f"poster cells embedded: {len(cell_ids)} in {time.time() - t0:.0f}s", flush=True)

# Real pieces, with one global white balance over all pieces (room light).
rows = list(csv.DictReader(open(os.path.join(pd, "patches.csv"))))
imgs = [cv2.imread(os.path.join(pd, f"p{r['id']}.png")) for r in rows]
masks = [cv2.imread(os.path.join(pd, f"m{r['id']}.png"), 0) for r in rows]
px = np.concatenate([im[mk > 128] for im, mk in zip(imgs, masks)]).astype(np.float64)
gains = px.mean() / px.mean(0)
print("white balance gains (B,G,R):", np.round(gains, 2))
pb = [prep(np.clip(im * gains, 0, 255).astype(np.uint8), mk) for im, mk in zip(imgs, masks)]
pc, pg = [], []
for i in range(0, len(pb), 64):
    a, b = embed(pb[i:i + 64]); pc.append(a); pg.append(b)
pc = torch.cat(pc); pg = torch.cat(pg)
print(f"pieces embedded in {time.time() - t0:.0f}s", flush=True)

dups = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))
        if rows[i]["photo"] != rows[j]["photo"] and math.hypot(float(rows[i]["mapx"]) - float(rows[j]["mapx"]), float(rows[i]["mapy"]) - float(rows[j]["mapy"])) < 25]

for name, P, C in (("CLS", pc, cells_cls), ("GRID4x4", pg, cells_grid), ("BOTH", torch.cat([pc, pg], 1) / math.sqrt(2), torch.cat([cells_cls, cells_grid], 1) / math.sqrt(2))):
    sim = P @ C.T                                   # (pieces, cells*rot)
    cell = cell_ids // 4
    # best per cell (over rotations)
    best_cell_sim = torch.full((len(rows), COLS * ROWS), -1.0)
    best_cell_sim.scatter_reduce_(1, cell.expand(len(rows), -1), sim, reduce="amax")
    top2 = best_cell_sim.topk(2, dim=1)
    pred = top2.indices[:, 0]; margin = (top2.values[:, 0] - top2.values[:, 1]).numpy()
    pc_ = (pred % COLS).numpy(); pr_ = (pred // COLS).numpy()
    conf = [i for i, r in enumerate(rows) if r["confident"] == "true"]
    agree = sum(abs(pc_[i] + 0.5 - float(rows[i]["col"])) < 0.6 and abs(pr_[i] + 0.5 - float(rows[i]["row"])) < 0.6 for i in conf)
    line = f"{name}: ncc_agree {agree}/{len(conf)}"
    for th in (0.0, 0.02, 0.04, 0.08):
        sel = [(i, j) for i, j in dups if margin[i] > th and margin[j] > th]
        same = sum(pc_[i] == pc_[j] and pr_[i] == pr_[j] for i, j in sel)
        line += f" | margin>{th}: cover {(margin > th).mean():.2f}, dup-agree {same}/{len(sel)}"
    print(line, flush=True)
    np.save(os.path.join(pd, f"dino_{name}.npy"), np.stack([pc_, pr_, margin]))
