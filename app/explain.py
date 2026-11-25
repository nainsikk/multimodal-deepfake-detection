import os
import uuid

import cv2
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEXT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME).eval()

image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()

HEATMAP_DIR = os.path.join("results", "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _text_token_weights(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]

    def forward(ids, mask):
        logits = text_model(input_ids=ids, attention_mask=mask).logits
        return torch.softmax(logits, dim=1)

    with torch.no_grad():
        target = int(forward(input_ids, attn).argmax(dim=1))

    baseline = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline[:, 0] = tokenizer.cls_token_id
    baseline[:, -1] = tokenizer.sep_token_id

    lig = LayerIntegratedGradients(forward, text_model.distilbert.embeddings)
    attrs = lig.attribute(
        input_ids, baselines=baseline,
        additional_forward_args=(attn,), target=target, n_steps=30,
    )
    scores = attrs.sum(dim=-1).squeeze(0)
    scores = scores / (scores.abs().max() + 1e-9)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return [
        {"token": t, "weight": round(float(w), 4)}
        for t, w in zip(tokens, scores)
        if t not in ("[CLS]", "[SEP]")
    ]


def _gradcam_heatmap(image_path):
    img = Image.open(image_path).convert("RGB")
    x = img_tf(img).unsqueeze(0)

    store = {}
    layer = image_model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))

    out = image_model(x)
    image_model.zero_grad()
    out[0, out.argmax(dim=1)].backward()
    h1.remove(); h2.remove()

    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["act"]).sum(dim=1)).squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    orig = np.array(img.resize((224, 224)))
    cam = cv2.resize(cam, (224, 224))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * orig + 0.5 * heat)

    out_path = os.path.join(HEATMAP_DIR, f"{uuid.uuid4().hex}.png")
    Image.fromarray(overlay).save(out_path)
    return out_path


def explain(text, image_path):
    return {
        "text_tokens": _text_token_weights(text),
        "image_heatmap_path": _gradcam_heatmap(image_path),
    }