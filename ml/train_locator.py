"""
Trains a per-puzzle placement model: straightened piece image (+ outline mask) -> (cell, rotation).

Training data is synthesised from the rectified box picture: random positions are cut out with random
jigsaw outlines and degraded like real camera photos (warm light, blur, noise, imperfect corners).
Evaluated on real pieces exported from the user's table photos.

Usage: python train_locator.py <patch_dir> <out_dir> [steps]
"""
import csv, math, os, random, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL = 64          # poster pixels per cell
WIN = 128          # piece window (2 cells), corner square = [32, 96]
INP = 64           # network input size

patch_dir, out_dir = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
os.makedirs(out_dir, exist_ok=True)
poster = cv2.imread(os.path.join(patch_dir, "poster64.png"))
PH, PW = poster.shape[:2]
COLS = PW // CELL
ROWS = math.ceil(PH / CELL)
NCLS = COLS * ROWS * 4
print(f"poster {PW}x{PH}, grid {COLS}x{ROWS}, classes {NCLS}", flush=True)
pad = cv2.copyMakeBorder(poster, WIN, WIN, WIN, WIN, cv2.BORDER_REFLECT)


def knob(side_len, flip, rng):
    """Classic jigsaw edge (3 cubic Beziers) in (u along, v across) units of side_len."""
    t = rng.uniform(0.095, 0.14); j = 0.03
    a, b, c, d, e = (rng.uniform(-j, j) for _ in range(5))
    P = [(0, 0), (0.2, a), (0.5 + b + d, -t + c), (0.5 - t + b, t + c), (0.5 - 2 * t + b - d, 3 * t + c),
         (0.5 + 2 * t + b - d, 3 * t + c), (0.5 + t + b, t + c), (0.5 + b + d, -t + c), (0.8, e), (1, 0)]
    pts = []
    for s in range(3):
        q = P[s * 3:s * 3 + 4]
        for k in range(0 if s == 0 else 1, 13):
            tt = k / 12; m = 1 - tt
            w = (m ** 3, 3 * m * m * tt, 3 * m * tt * tt, tt ** 3)
            pts.append((sum(w[i] * q[i][0] for i in range(4)), flip * sum(w[i] * q[i][1] for i in range(4))))
    return pts


def piece_mask(rng, flat):
    """Mask of a random piece in the 128 window; flat[side] for border sides (top,right,bottom,left)."""
    corners = [(32, 32), (96, 32), (96, 96), (32, 96)]
    poly = []
    for s in range(4):
        (x0, y0), (x1, y1) = corners[s], corners[(s + 1) % 4]
        dx, dy = x1 - x0, y1 - y0
        if flat[s]:
            pts = [(0, 0), (1, 0)]
        else:
            pts = knob(64, rng.choice([-1, 1]), rng)
        for (u, v) in pts[:-1]:
            # normal (-dy, dx) points into the square for clockwise order; v<0 = outward (tab)
            poly.append((x0 + u * dx - v * dy, y0 + u * dy + v * dx))
    m = np.zeros((WIN, WIN), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 255)
    return m


def degrade(img, rng):
    img = img.astype(np.float32)
    # warm room light / white-balance error + brightness + gamma
    gains = np.array([rng.uniform(0.55, 1.1), rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.5)], np.float32)  # B,G,R
    img = img * gains * rng.uniform(0.55, 1.35)
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rng.uniform(0.8, 1.25)
    # soft shading gradient
    gx, gy = np.meshgrid(np.linspace(-1, 1, WIN), np.linspace(-1, 1, WIN))
    img = img * (1 + rng.uniform(-0.15, 0.15) * gx + rng.uniform(-0.15, 0.15) * gy)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:  # washed-out / desaturated photo
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
        a = rng.uniform(0.4, 1.0)
        img = np.clip(img * a + grey * (1 - a) + rng.uniform(0, 30), 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.6))
    if rng.random() < 0.7:
        img = np.clip(img.astype(np.int16) + np.random.normal(0, rng.uniform(1, 7), img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def to_input(img, mask):
    """128x128 BGR + mask -> (4, 64, 64) float tensor; outside the mask = grey."""
    img = img.copy(); img[mask < 128] = 128
    x = cv2.resize(img, (INP, INP), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - 0.5
    m = cv2.resize(mask, (INP, INP), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - 0.5
    return np.concatenate([x.transpose(2, 0, 1), m[None]], 0)


def sample(rng):
    cx = rng.uniform(CELL / 2, PW - CELL / 2); cy = rng.uniform(CELL / 2, PH - CELL / 2)
    col = min(COLS - 1, int(cx // CELL)); row = min(ROWS - 1, int(cy // CELL))
    # snap mostly to the cell centre (real pieces sit on the grid), with jitter
    cx = (col + 0.5) * CELL + rng.gauss(0, 4); cy = (row + 0.5) * CELL + rng.gauss(0, 4)
    flat = [row == 0, col == COLS - 1, row == ROWS - 1, col == 0]
    x0 = int(round(cx - WIN / 2)) + WIN; y0 = int(round(cy - WIN / 2)) + WIN
    win = pad[y0:y0 + WIN, x0:x0 + WIN]
    mask = piece_mask(rng, flat)
    # imperfect straightening: random perspective jitter of the corner square
    src = np.float32([[32, 32], [96, 32], [96, 96], [32, 96]])
    dst = src + np.float32([[rng.gauss(0, 2.5), rng.gauss(0, 2.5)] for _ in range(4)])
    s = rng.uniform(0.92, 1.08); ang = math.radians(rng.uniform(-4, 4))
    R = np.float32([[s * math.cos(ang), -s * math.sin(ang)], [s * math.sin(ang), s * math.cos(ang)]])
    dst = (dst - 64) @ R.T + 64
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    win = cv2.warpPerspective(win, H, (WIN, WIN), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpPerspective(mask, H, (WIN, WIN))
    if rng.random() < 0.5:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=rng.randint(1, 3))
    # imperfect real outlines: random bites taken out / blobs of neighbour added
    for _ in range(rng.choice([0, 0, 1, 2, 3])):
        cx_, cy_ = rng.randint(16, 112), rng.randint(16, 112)
        cv2.circle(mask, (cx_, cy_), rng.randint(6, 22), 0 if rng.random() < 0.6 else 255, -1)
    img = degrade(win, rng)
    k = rng.randint(0, 3)  # patch = poster rotated k quarter turns clockwise
    for _ in range(k):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE); mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    return to_input(img, mask), (row * COLS + col) * 4 + k


class Synth(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random((info.id if info else 0) * 7919 + int(time.time()))
        np.random.seed(rng.randint(0, 2 ** 31))
        while True:
            x, y = sample(rng)
            yield torch.from_numpy(x), y


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(block(4, 48, 2), block(48, 96, 2), block(96, 192, 2), block(192, 256, 2))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, NCLS))

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.f(x), 1).flatten(1))


# ---- real evaluation set ----
real = []
with open(os.path.join(patch_dir, "patches.csv")) as fh:
    for r in csv.DictReader(fh):
        img = cv2.imread(os.path.join(patch_dir, f"p{r['id']}.png")); m = cv2.imread(os.path.join(patch_dir, f"m{r['id']}.png"), 0)
        real.append((r, torch.from_numpy(to_input(img, m))))
real_x = torch.stack([t for _, t in real])
dups = []  # same physical piece seen in two photos
for i in range(len(real)):
    for j in range(i + 1, len(real)):
        a, b = real[i][0], real[j][0]
        if a["photo"] != b["photo"] and math.hypot(float(a["mapx"]) - float(b["mapx"]), float(a["mapy"]) - float(b["mapy"])) < 25:
            dups.append((i, j))
print(f"real patches {len(real)}, ncc-confident {sum(r['confident'] == 'true' for r, _ in real)}, duplicate pairs {len(dups)}", flush=True)


def evaluate(net):
    net.eval()
    with torch.no_grad():
        p = F.softmax(net(real_x), 1)
        # marginalise rotation for the cell decision
        pc = p.view(len(real), ROWS * COLS, 4).sum(2)
        conf, cell = pc.max(1)
    net.train()
    cols_ = (cell % COLS).numpy(); rows_ = (cell // COLS).numpy(); conf = conf.numpy()
    agree_ncc = [abs(cols_[i] + 0.5 - float(r["col"])) < 0.6 and abs(rows_[i] + 0.5 - float(r["row"])) < 0.6
                 for i, (r, _) in enumerate(real) if r["confident"] == "true"]
    out = {"ncc_agree": f"{sum(agree_ncc)}/{len(agree_ncc)}"}
    for th in (0.3, 0.5, 0.7):
        sel = [(i, j) for i, j in dups if conf[i] > th and conf[j] > th]
        same = sum(cols_[i] == cols_[j] and rows_[i] == rows_[j] for i, j in sel)
        out[f"dup@{th}"] = f"{same}/{len(sel)}"
        out[f"cov@{th}"] = f"{(conf > th).mean():.2f}"
    allsame = sum(cols_[i] == cols_[j] and rows_[i] == rows_[j] for i, j in dups)
    out["dup_all"] = f"{allsame}/{len(dups)}"
    return out, conf, cols_, rows_


torch.set_num_threads(12)
net = Net()
print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.1)
dl = torch.utils.data.DataLoader(Synth(), batch_size=192, num_workers=8)
t0 = time.time(); run_loss = 0; run_acc = 0
for step, (x, y) in enumerate(dl):
    if step >= STEPS:
        break
    logits = net(x)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    run_loss += loss.item(); run_acc += (logits.argmax(1) == y).float().mean().item()
    if (step + 1) % 250 == 0:
        ev, *_ = evaluate(net)
        print(f"step {step + 1} loss {run_loss / 250:.3f} synth-acc {run_acc / 250:.3f} | real {ev} | {time.time() - t0:.0f}s", flush=True)
        run_loss = run_acc = 0
        torch.save(net.state_dict(), os.path.join(out_dir, "locator.pt"))

net.eval()
torch.save(net.state_dict(), os.path.join(out_dir, "locator.pt"))
torch.onnx.export(net, torch.zeros(1, 4, INP, INP), os.path.join(out_dir, "locator.onnx"), input_names=["input"], output_names=["logits"], opset_version=13, dynamo=False)
with open(os.path.join(out_dir, "locator.txt"), "w") as fh:
    fh.write(f"cols {COLS}\nrows {ROWS}\ncell {CELL}\ninput {INP}\n")
ev, *_ = evaluate(net)
print("FINAL", ev, flush=True)
