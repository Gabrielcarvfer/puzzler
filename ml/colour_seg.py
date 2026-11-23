"""Prototype: (1) table colour model by histogram back-projection, (2) black-hat to cut touching pieces along their shadow lines."""
import sys, glob, cv2, numpy as np
fs = sorted(glob.glob('session/photos/*.jpg')); f = fs[int(sys.argv[1]) if len(sys.argv) > 1 else 12]
im = cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_2)
rot = int(open(f[:-4] + '.txt').read().split()[1])
im = {90: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), 270: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE), 180: lambda x: cv2.rotate(x, cv2.ROTATE_180)}.get(rot, lambda x: x)(im)
H, W = im.shape[:2]
blur = cv2.GaussianBlur(im, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
# --- table sample: smooth pixels whose hue/sat is the image's dominant mode ---
L = lab[..., 0].astype(np.float32)
grad = cv2.blur(cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)), (9, 9))
smooth = grad < np.percentile(grad, 35)
hist_all = cv2.calcHist([hsv], [0, 1], smooth.astype(np.uint8) * 255, [45, 32], [0, 180, 0, 256])
h0, s0 = np.unravel_index(hist_all.argmax(), hist_all.shape)
near = (np.abs((hsv[..., 0] // 4).astype(int) - h0) <= 2) & (np.abs((hsv[..., 1] // 8).astype(int) - s0) <= 6) & smooth
print(f"{f.split('/')[-1]}: table mode hue {h0*4} sat {s0*8}, sample {near.mean()*100:.0f}% of image")
# --- (1) back-projection of the table's hue/saturation histogram ---
hist = cv2.calcHist([hsv], [0, 1], near.astype(np.uint8) * 255, [45, 32], [0, 180, 0, 256])
hist = cv2.GaussianBlur(hist, (5, 5), 0); cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
bp = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
bp = cv2.GaussianBlur(bp, (7, 7), 0)
_, table = cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mask = cv2.bitwise_not(table)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
# --- (2) black-hat: thin dark lines (shadow between touching pieces) ---
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
bh = cv2.morphologyEx(lab[..., 0], cv2.MORPH_BLACKHAT, k)
lines = (bh > 28).astype(np.uint8) * 255
lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
cut = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.dilate(lines, np.ones((3, 3), np.uint8))))
cut = cv2.morphologyEx(cut, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
def blobs(m):
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, 8)
    a = np.sqrt(st[1:, cv2.CC_STAT_AREA]); a = a[a > 40]
    med = np.median(a) if len(a) else 0
    return n, lab_, st, int(((a > 0.7 * med) & (a < 1.4 * med)).sum()), med
for name, m in (("back-projection mask", mask), ("+ black-hat cuts", cut)):
    n, _, _, good, med = blobs(m); print(f"  {name}: {good} piece-sized blobs (typical size {med:.0f} px)")
n, lb, st, _, med = blobs(cut)
vis = np.zeros_like(im); rng = np.random.RandomState(1)
for i in range(1, n):
    a = np.sqrt(st[i, cv2.CC_STAT_AREA])
    if a < 40: continue
    ok = 0.7 * med < a < 1.4 * med
    vis[lb == i] = (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255)) if ok else (70, 70, 70)
out = np.hstack([im, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), vis])
cv2.imwrite('test/colour_seg.jpg', cv2.resize(out, None, fx=0.43, fy=0.43))
