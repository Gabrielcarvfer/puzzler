import sys, numpy as np, torch, cv2, onnxruntime as ort
from transformers import DetrForObjectDetection
from onnxruntime.quantization import quantize_dynamic, QuantType
name = "pablo-moreira/puzzle-piece-detection"; out = sys.argv[1]; N = 640
m = DetrForObjectDetection.from_pretrained(name).eval()
print("labels", m.config.id2label)
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        o = s.m(pixel_values=x, pixel_mask=torch.ones(x.shape[0], x.shape[2], x.shape[3], dtype=torch.long))
        return o.logits.softmax(-1)[..., 0], o.pred_boxes
x = torch.randn(1, 3, N, N)
torch.onnx.export(W(m), x, out + "/detr_fp32.onnx", input_names=["pixels"], output_names=["scores", "boxes"], opset_version=17, dynamo=False)
quantize_dynamic(out + "/detr_fp32.onnx", out + "/detr.onnx", weight_type=QuantType.QUInt8)
# verify on a real tile
img = cv2.imread(sys.argv[2], cv2.IMREAD_REDUCED_COLOR_2)[1500:2700, 1000:2200]
rgb = cv2.cvtColor(cv2.resize(img, (N, N)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
inp = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
with torch.no_grad(): ts, tb = W(m)(torch.from_numpy(inp))
for f in ("detr_fp32.onnx", "detr.onnx"):
    s, b = ort.InferenceSession(out + "/" + f).run(None, {"pixels": inp})
    print(f, "pieces>0.5:", int((s > 0.5).sum()), "torch:", int((ts > 0.5).sum()), "max box diff vs torch", float(np.abs(b - tb.numpy())[s > 0.5].max()) if (s > 0.5).any() else 0)
