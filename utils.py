import torch
import cv2

# Characters
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

idx_to_char = {i+1: c for i, c in enumerate(CHARS)}

# ✅ PREPROCESS FUNCTION
def preprocess(img):
    img = cv2.resize(img, (128, 32))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img

# ✅ FINAL DECODE FUNCTION (CTC CLEANED)
def decode(pred):
    pred = pred.argmax(2)
    result = ""
    prev = -1

    for p in pred[0]:
        if p.item() != prev and p.item() != 0:
            char = idx_to_char.get(p.item(), "")
            result += char
        prev = p.item()

    # ✅ remove non-valid patterns (simple filter)
    return result[:10]   # limit length