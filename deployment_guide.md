# PetVision AI - Free Cloud Deployment Guide 🚀

This guide provides exact, step-by-step instructions for deploying your **PetVision AI** web application for free so that you can place the live public URL at the very top of your Jupyter Notebook for submission.

---

## Option 1: Hugging Face Spaces (Recommended ⭐️)
*Why recommended?* Hugging Face Spaces provides free 16 GB RAM CPU hosting, handles PyTorch model weights easily, and gives an instant public `.hf.space` link without Git LFS complications.

### Step 1: Create a Free Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co) and sign up (or log in).
2. Click on your profile icon (top-right) and select **New Space**.

### Step 2: Configure Your Space
- **Space Name:** `petvision-ai`
- **License:** `mit` or `openrail`
- **Select the Space SDK:** Choose **Streamlit**
- **Space Hardware:** Select **Free (CPU basic · 2 vCPU · 16 GB RAM)**
- **Visibility:** Public
- Click **Create Space**.

### Step 3: Upload Files
In your new Space repository, click **Files** -> **Add file** -> **Upload files**:
1. Upload `app/app.py` directly to the root as `app.py`.
2. Upload `app/requirements.txt` to the root as `requirements.txt`.
3. Create a folder `results/` and upload `results/class_mappings.json`.
4. Upload `models/best_model.pth` (or git push the repository).
5. Hugging Face will automatically build and launch your Streamlit app in ~2 minutes!
6. Copy your live Space URL (e.g., `https://huggingface.co/spaces/<your-username>/petvision-ai`) and paste it at the top of `notebooks/PetVision_AI.ipynb`.

---

## Option 2: Streamlit Community Cloud (via GitHub)

### Step 1: Push Project to GitHub
1. Create a new GitHub repository named `PetVision-AI` at [github.com/new](https://github.com/new).
2. In your local terminal, run:
```bash
git init
git add .
git commit -m "Initial PetVision AI commit"
git branch -M main
git remote add origin https://github.com/<your-username>/PetVision-AI.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **Create app**.
3. Select your repository: `<your-username>/PetVision-AI`.
4. Set **Main file path** to: `app/app.py`.
5. Click **Deploy!**.
6. Once deployed, copy your URL (e.g., `https://petvision-ai.streamlit.app`) and paste it at the very top of `PetVision_AI.ipynb`.
