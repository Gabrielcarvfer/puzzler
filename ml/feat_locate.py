"""
Segmentation-free piece location: SIFT features of a camera frame matched against the box picture,
grouped into pieces by geometric consistency (similarity transform per piece).
Usage: python feat_locate.py <box.jpg> <frame.jpg> <out.jpg> [x0 y0 w h]
"""
import sys, time, math, cv2, numpy as np
COLS, ROWS, CELL = 38, 26, 128
box = cv2.imread(sys.argv[1])
q = np.float32([[254, 430], [7954, 410], [7980, 5948], [238, 5932]])
W = COLS * CELL; H = int(round(W * 480 / 683))
M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
poster = cv2.warpPerspective(box, M, (W, H))
frame = cv2.imread(sys.argv[2])
if len(sys.argv) > 4:
    x0, y0, w, h = map(int, sys.argv[4:8]); frame = frame[y0:y0 + h, x0:x0 + w]
import os
SC = float(os.environ.get('SCALE', '1'))
if SC != 1: frame = cv2.resize(frame, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)
sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02)
t = time.time()
pk, pd = sift.detectAndCompute(cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY), None)
t1 = time.time()
fk, fd = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
t2 = time.time()
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
knn = flann.knnMatch(fd, pd, k=2)
good = [a for a, b in knn if a.distance < 0.8 * b.distance]
t3 = time.time()
print(f"poster {W}x{H}: {len(pk)} kp ({t1 - t:.1f}s) | frame {frame.shape[1]}x{frame.shape[0]}: {len(fk)} kp ({t2 - t1:.1f}s) | {len(good)} ratio-test matches ({t3 - t2:.1f}s)")
F = np.float32([fk[m.queryIdx].pt for m in good]); P = np.float32([pk[m.trainIdx].pt for m in good])
# scale (frame px per poster px) implied by each match
S = np.float32([fk[m.queryIdx].size / pk[m.trainIdx].size for m in good])
used = np.zeros(len(good), bool); pieces = []
order = np.argsort([m.distance for m in good])
for i in order:
    if used[i]: continue
    near = (~used) & (np.hypot(*(P - P[i]).T) < CELL * 1.0)
    idx = np.where(near)[0]
    if len(idx) < 4: continue
    s = float(np.median(S[idx])); 
    idx = idx[np.hypot(*(F[idx] - F[i]).T) < CELL * s * 1.6]
    if len(idx) < 4: continue
    A, inl = cv2.estimateAffinePartial2D(P[idx], F[idx], method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if A is None: continue
    inl = inl.ravel().astype(bool)
    if inl.sum() < 4: continue
    sc = math.hypot(A[0, 0], A[1, 0]); ang = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    ip = P[idx][inl]
    # inliers must spread over the piece, not sit on one spot
    spread = np.hypot(*(ip - ip.mean(0)).T).max()
    if spread < CELL * 0.15: continue
    used[idx[inl]] = True
    # cell = the grid cell holding most inliers (tabs reach into neighbours, so not the centroid)
    cc = np.floor(ip / CELL).astype(int); u, cnt = np.unique(cc, axis=0, return_counts=True)
    c = (u[cnt.argmax()] + 0.5) * CELL
    pieces.append(dict(n=int(inl.sum()), scale=sc, ang=ang, col=c[0] / CELL, row=c[1] / CELL, A=A, fx=float(F[idx][inl][:, 0].mean()), fy=float(F[idx][inl][:, 1].mean())))
if not pieces: print("no pieces"); sys.exit()
med = np.median([p["scale"] for p in pieces if p["n"] >= 6] or [p["scale"] for p in pieces])
ok = [p for p in pieces if abs(math.log(p["scale"] / med)) < 0.18]
print(f"groups {len(pieces)}, median scale {med:.2f}, scale-consistent {len(ok)}; by inliers: " + ", ".join(f">={k}: {sum(p['n'] >= k for p in ok)}" for k in (4, 5, 6, 8, 12)))
# merge groups that are the same piece (same cell + close in frame)
ok.sort(key=lambda p: -p["n"]); final = []
for p in ok:
    if any(math.hypot(p["fx"] - o["fx"], p["fy"] - o["fy"]) < CELL * med * 0.6 for o in final): continue
    final.append(p)
print(f"distinct pieces located: {len(final)} (>=5 inliers: {sum(p['n'] >= 5 for p in final)}, >=6: {sum(p['n'] >= 6 for p in final)})")
cells = {}
for p in final:
    if p["n"] >= 5: cells[(int(p["col"]), int(p["row"]))] = p
pairs = [(a, b) for a in cells for b in cells if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
print(f"pairs (adjacent cells, both >=5 inliers): {len(pairs)}")
vis = frame.copy()
for p in final:
    if p["n"] < 5: continue
    col, row = int(p["col"]), int(p["row"])
    quad = np.float32([[col * CELL, row * CELL], [(col + 1) * CELL, row * CELL], [(col + 1) * CELL, (row + 1) * CELL], [col * CELL, (row + 1) * CELL]])
    fq = (quad @ p["A"][:, :2].T + p["A"][:, 2]).astype(np.int32)
    cv2.polylines(vis, [fq], True, (0, 255, 0) if p["n"] >= 8 else (0, 200, 255), 4)
    cv2.putText(vis, f"{row + 1}.{col + 1}", (int(p["fx"]) - 40, int(p["fy"])), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
for a, b in pairs:
    pa, pb = cells[a], cells[b]
    cv2.line(vis, (int(pa["fx"]), int(pa["fy"])), (int(pb["fx"]), int(pb["fy"])), (255, 0, 255), 6)
cv2.imwrite(sys.argv[3], vis)
np.save(sys.argv[3] + ".npy", np.array([[p["fx"] / SC, p["fy"] / SC, int(p["col"]), int(p["row"]), p["n"], p["ang"]] for p in final]))
