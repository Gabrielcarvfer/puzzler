"""Feasibility: is the height of the pieces (about 2 mm) visible as parallax between two sweep photos?"""
import sys, glob, cv2, numpy as np
fs = sorted(glob.glob('session/photos/*.jpg'))
i = int(sys.argv[1]) if len(sys.argv) > 1 else 12
step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
a = cv2.imread(fs[i], cv2.IMREAD_REDUCED_COLOR_2); b = cv2.imread(fs[i + step], cv2.IMREAD_REDUCED_COLOR_2)
ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(8000)
ka, da = sift.detectAndCompute(ga, None); kb, db = sift.detectAndCompute(gb, None)
m = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.7 * y.distance]
pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
H1, in1 = cv2.findHomography(pb, pa, cv2.RANSAC, 1.5); in1 = in1.ravel().astype(bool)
print(f"{fs[i].split('/')[-1]} vs {fs[i+step].split('/')[-1]}: {len(m)} matches; plane 1 fits {in1.sum()}")
shift = np.linalg.norm((pa - pb)[in1], axis=1)
print(f"  camera moved the image by about {np.median(shift):.0f} px between the two photos")
rest = ~in1
if rest.sum() >= 12:
    H2, in2 = cv2.findHomography(pb[rest], pa[rest], cv2.RANSAC, 1.5); in2 = in2.ravel().astype(bool)
    print(f"  second plane fits {in2.sum()} of the remaining {rest.sum()} matches")
    # where plane-2 points land under plane-1's mapping vs where they really are = parallax between planes
    q = cv2.perspectiveTransform(pb[rest][in2][None], H1)[0]
    par = np.linalg.norm(q - pa[rest][in2], axis=1)
    print(f"  parallax between the two planes: median {np.median(par):.1f} px (25%-75%: {np.percentile(par,25):.1f}-{np.percentile(par,75):.1f}) at half resolution")
# residual of every match under plane 1 (0 = on plane 1)
res = np.linalg.norm(cv2.perspectiveTransform(pb[None], H1)[0] - pa, axis=1)
print("  residuals of all matches under plane 1 (px): " + " ".join(f"{p}%:{np.percentile(res,p):.1f}" for p in (50, 75, 90, 95, 99)))
# dense: align b onto a with plane 1, then optical flow = what is NOT on plane 1
wb = cv2.warpPerspective(b, H1, (a.shape[1], a.shape[0]))
flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ga, cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY), None)
mag = np.linalg.norm(flow, axis=2); valid = cv2.warpPerspective(np.full(gb.shape, 255, np.uint8), H1, (a.shape[1], a.shape[0])) > 0
mag[~valid] = 0
print(f"  dense flow after aligning plane 1: median {np.median(mag[valid]):.2f} px, 90%: {np.percentile(mag[valid],90):.2f}, 99%: {np.percentile(mag[valid],99):.2f}")
vis = cv2.applyColorMap(np.clip(mag * 255 / 8, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
pts = a.copy()
for (x, y), r in zip(pa, res):
    cv2.circle(pts, (int(x), int(y)), 6, (0, 255, 0) if r < 1.5 else (0, 0, 255) if r > 3 else (0, 255, 255), -1)
out = np.hstack([pts, vis])
cv2.imwrite('test/parallax.jpg', cv2.resize(out, None, fx=0.5, fy=0.5))
