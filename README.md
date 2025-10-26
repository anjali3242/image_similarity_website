# Image Similarity Checker

A **FastAPI** web application that allows users to **compare two images** and measure their similarity using multiple metrics:

- **CLIP Similarity** – Semantic similarity based on OpenAI's CLIP model  
- **MSE (Mean Squared Error)** – Pixel-level difference  
- **SSIM (Structural Similarity)** – Structural similarity between images  

The app provides an **interactive web interface** with image previews, result explanations, and a clean, responsive design.

---

## **Features**

- Upload two images via the web interface.  
- Compute and display similarity scores: **CLIP, MSE, SSIM**.  
- Show uploaded image previews before comparison.  
- Detailed explanation for each similarity metric.  
- Responsive design using **Bootstrap 5**.  
- Works locally or can be deployed online (Render, Replit, etc.).  

---

## **Demo Screenshot**

![Demo Screenshot](demo_screenshot.png)(http://127.0.0.1:8000/upload)

*(Add a screenshot of your app here)*

---

## **Requirements**

- Python 3.10+  
- Packages listed in `requirements.txt`:

