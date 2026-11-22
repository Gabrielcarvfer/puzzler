import sys, time, cv2, torch
from transformers import DetrForObjectDetection, DetrImageProcessor
torch.set_num_threads(16)
name = "pablo-moreira/puzzle-piece-detection"
proc = DetrImageProcessor.from_pretrained(name); model = DetrForObjectDetection.from_pretrained(name).eval()
for path, out, tiles in [(a, b, int(c)) for a, b, c in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3])]:
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    H, W = img.shape[:2]; n = 0; t = time.time()
    # tile so each tile holds well under DETR's 100-query limit
    for ty in range(tiles):
        for tx in range(tiles):
            x0, y0 = W * tx // tiles, H * ty // tiles; x1, y1 = W * (tx + 1) // tiles, H * (ty + 1) // tiles
            crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                o = model(**proc(images=crop, return_tensors="pt"))
            r = proc.post_process_object_detection(o, threshold=0.5, target_sizes=[crop.shape[:2]])[0]
            for s, b in zip(r["scores"], r["boxes"]):
                b = b.int().tolist(); n += 1
                cv2.rectangle(img, (x0 + b[0], y0 + b[1]), (x0 + b[2], y0 + b[3]), (0, 255, 0), 3)
    cv2.imwrite(out, img); print(f"{path.split('/')[-1]}: {n} pieces, {tiles}x{tiles} tiles, {time.time()-t:.1f}s", flush=True)
