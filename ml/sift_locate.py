"""SIFT placement (as in whitcrrd/puzzle-image-segmentation): piece patch vs. box picture, homography -> cell."""
import csv, math, os, sys, cv2, numpy as np
pd = sys.argv[1]
poster = cv2.imread(os.path.join(pd, "poster64.png")); C = 64; COLS = poster.shape[1] // C
sift = cv2.SIFT_create()
pk, pdsc = sift.detectAndCompute(cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY), None)
bf = cv2.BFMatcher(cv2.NORM_L2)
rows = list(csv.DictReader(open(os.path.join(pd, "patches.csv"))))
res = []
for r in rows:
    img = cv2.imread(os.path.join(pd, f"p{r['id']}.png")); m = cv2.imread(os.path.join(pd, f"m{r['id']}.png"), 0)
    k, d = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), m)
    out = None
    if d is not None and len(k) >= 4:
        good = [a for a, b in (x for x in bf.knnMatch(d, pdsc, k=2) if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(good) >= 4:
            src = np.float32([k[g.queryIdx].pt for g in good]); dst = np.float32([pk[g.trainIdx].pt for g in good])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
            if H is not None:
                inl = int(mask.sum()); c = cv2.perspectiveTransform(np.float32([[[64, 64]]]), H)[0, 0]
                out = (c[0] / C, c[1] / C, inl, len(k))
    res.append(out)
n_kp = [o[3] for o in res if o]
print(f"pieces {len(rows)}, with a homography {sum(o is not None for o in res)}, median keypoints/piece {np.median(n_kp) if n_kp else 0}")
dups = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if rows[i]["photo"] != rows[j]["photo"] and math.hypot(float(rows[i]["mapx"]) - float(rows[j]["mapx"]), float(rows[i]["mapy"]) - float(rows[j]["mapy"])) < 25]
for th in (4, 6, 8, 12):
    ok = [i for i, o in enumerate(res) if o and o[2] >= th]
    conf = [i for i in ok if rows[i]["confident"] == "true"]
    agree = sum(abs(res[i][0] - float(rows[i]["col"])) < 0.7 and abs(res[i][1] - float(rows[i]["row"])) < 0.7 for i in conf)
    sel = [(i, j) for i, j in dups if i in ok and j in ok]
    same = sum(math.hypot(res[i][0] - res[j][0], res[i][1] - res[j][1]) < 0.7 for i, j in sel)
    print(f"inliers>={th}: placed {len(ok)}/{len(rows)}, agrees with NCC-confident {agree}/{len(conf)}, same piece twice -> same spot {same}/{len(sel)}")
