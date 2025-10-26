from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import clip
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

templates = Jinja2Templates(directory="template")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_similarity(image1: Image.Image, image2: Image.Image):
    image1_input = preprocess(image1).unsqueeze(0).to(device)
    image2_input = preprocess(image2).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = model.encode_image(image1_input)
        emb2 = model.encode_image(image2_input)
    emb1 /= emb1.norm(dim=-1, keepdim=True)
    emb2 /= emb2.norm(dim=-1, keepdim=True)
    similarity = (emb1 @ emb2.T).item()
    return round(similarity * 100, 2)

def compute_mse(image1: Image.Image, image2: Image.Image):
    img1 = np.array(image1.resize((256, 256))).astype(np.float32)
    img2 = np.array(image2.resize((256, 256))).astype(np.float32)
    mse_val = np.mean((img1 - img2) ** 2)
    return round(mse_val, 2)

def compute_ssim(image1: Image.Image, image2: Image.Image):
    img1 = cv2.cvtColor(np.array(image1.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.array(image2.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(img1, img2)
    return round(ssim_val * 100, 2)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1 = Image.open(image1.file).convert("RGB")
        img2 = Image.open(image2.file).convert("RGB")

        clip_score = compute_clip_similarity(img1, img2)
        mse_score = compute_mse(img1, img2)
        ssim_score = compute_ssim(img1, img2)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "clip_score": clip_score,
                "mse_score": mse_score,
                "ssim_score": ssim_score,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e)}
        )
