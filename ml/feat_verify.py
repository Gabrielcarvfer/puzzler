import sys, math, cv2, numpy as np, itertools
R = "core/src/test/resources/map/IMG_20261122_%s.jpg"
ids = sys.argv[1:]
orb = cv2.ORB_create(8000); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
small = {i: cv2.imread(R % i, cv2.IMREAD_REDUCED_GRAYSCALE_4) for i in ids}
feats = {i: orb.detectAndCompute(small[i], None) for i in ids}
res = {i: np.load(f"test/feat_{i}.jpg.npy") for i in ids}
for a, b in itertools.combinations(ids, 2):
    (ka, da), (kb, db) = feats[a], feats[b]
    good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 30: print(a, b, "no overlap"); continue
    H, mask = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]) * 4, np.float32([kb[m.trainIdx].pt for m in good]) * 4, cv2.RANSAC, 12.0)
    if H is None or mask.sum() < 40: print(a, b, "no overlap"); continue
    pa = res[a]; pb = res[b]
    proj = cv2.perspectiveTransform(pa[:, :2].astype(np.float32)[None], H)[0]
    for th in (4, 5, 6, 8):
        tot = same = near = 0
        for (x, y), ra in zip(proj, pa):
            if ra[4] < th: continue
            d = np.hypot(pb[:, 0] - x, pb[:, 1] - y); j = d.argmin()
            if d[j] > 110 or pb[j, 4] < th: continue
            tot += 1; dd = abs(ra[2] - pb[j, 2]) + abs(ra[3] - pb[j, 3]); same += dd == 0; near += dd == 1
        print(f"{a} vs {b} (inliers {int(mask.sum())}): pieces seen in both with >={th} matches: {tot}, same cell: {same}, neighbour cell: {near}, elsewhere: {tot - same - near}")
