"""Prototype 2: table = pixels in the wood's hue band (any brightness), pieces = everything else."""
import sys, glob, cv2, numpy as np
fs = sorted(glob.glob('session/photos/*.jpg'))
tot_a = tot_b = 0
for idx in [int(a) for a in sys.argv[1:]] or [12]:
    f = fs[idx]
    im = cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_2)
    rot = int(open(f[:-4] + '.txt').read().split()[1])
    if rot == 90: im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 270: im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    hsv = cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(np.int32) for i in range(3)]
    hh = np.bincount(h[s > 120].ravel(), minlength=180)
    h0 = int(np.argmax(np.convolve(hh, np.ones(5), 'same')))
    dh = np.minimum(np.abs(h - h0), 180 - np.abs(h - h0))
    # wood: hue within the band; saturation may drop in glare but brightness then rises
    table = ((dh <= 7) & (s > 110)) | ((dh <= 10) & (s > 60) & (v > 200))
    mask = (~table).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lb, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    a = np.sqrt(st[1:, cv2.CC_STAT_AREA]); big = a[a > 60]
    # piece-sized = within the densest size cluster
    srt = np.sort(big); best = (0, 0)
    for i, r in enumerate(srt):
        cnt = int(((srt >= r) & (srt <= r * 1.35)).sum())
        if cnt * r > best[0]: best = (cnt * r, r * 1.15)
    size = best[1]; good = int(((big > 0.8 * size) & (big < 1.25 * size)).sum())
    print(f"{f.split('/')[-1]}: wood hue {h0*2} deg, piece-sized blobs {good} (size {size:.0f} px), bigger blobs (clumps/strips) {int((big > 1.25*size).sum())}")
    tot_a += good
    if idx == ([int(a) for a in sys.argv[1:]] or [12])[0]:
        vis = im.copy()
        for i in range(1, n):
            r = np.sqrt(st[i, cv2.CC_STAT_AREA])
            if r < 60: continue
            c, _ = cv2.findContours((lb == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, c, -1, (0, 255, 0) if 0.8 * size < r < 1.25 * size else (0, 0, 255), 4)
        cv2.imwrite('test/hue_seg.jpg', cv2.resize(np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]), None, fx=0.5, fy=0.5))
print("total piece-sized blobs:", tot_a)
