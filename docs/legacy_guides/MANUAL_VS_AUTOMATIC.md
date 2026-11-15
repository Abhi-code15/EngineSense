# Manual vs Automatic Uploads - Quick Guide

## 🚀 AUTOMATIC Uploads (Via Scripts)

These happen automatically when you run the scripts. **You don't need to manually upload anything to Hugging Face.**

### ✅ Automatic: Hugging Face Dataset Repo
**Script**: `python src/data_register.py` and `python src/data_prep.py`

**What gets uploaded automatically:**
- ✅ `data/engine_data.csv` → HF Dataset Repo
- ✅ `data/processed/train.csv` → HF Dataset Repo  
- ✅ `data/processed/test.csv` → HF Dataset Repo

**You just run:**
```bash
python src/data_register.py    # Auto-uploads raw data
python src/data_prep.py        # Auto-uploads train/test
```

**That's it!** The scripts handle everything - creating the repo, uploading files, etc.

---

### ✅ Automatic: Hugging Face Model Repo
**Script**: `python src/train.py`

**What gets uploaded automatically:**
- ✅ `models/best_model.joblib` → HF Model Repo (as `model.joblib`)

**You just run:**
```bash
python src/train.py
```

**That's it!** The script:
1. Trains the model
2. Saves it locally
3. Automatically uploads to HF Model Repo

---

### ✅ Automatic: Hugging Face Space (Streamlit App)
**Script**: `python src/deploy_to_hf.py`

**What gets uploaded automatically:**
- ✅ `src/app.py` (Streamlit app)
- ✅ `src/inference.py`
- ✅ `src/config.py`
- ✅ `Dockerfile`
- ✅ `requirements.txt`
- ✅ Other `src/*.py` files needed

**You just run:**
```bash
python src/deploy_to_hf.py
```

**That's it!** The script:
1. Creates/updates the HF Space
2. Uploads all deployment files
3. Configures it as a Streamlit app

---

## 📤 MANUAL Uploads (You Do This)

Only **ONE** thing needs to be done manually:

### ⚠️ Manual: GitHub Repository

**What you need to do manually:**
- Push your entire `mlops/` folder to GitHub

**Steps:**
```bash
cd /Users/Abhishektripathi/Desktop/mlops

# 1. Initialize git (if not done)
git init

# 2. Create .gitignore (to exclude large files)
cat > .gitignore << EOF
.venv/
__pycache__/
*.pyc
mlruns/
*.log
.DS_Store
EOF

# 3. Add all files
git add .

# 4. Commit
git commit -m "Initial commit: Predictive Maintenance MLOps Pipeline"

# 5. Add your GitHub repo as remote
git remote add origin https://github.com/Abhishektripathi/engine-predictive-maintenance.git

# 6. Push to GitHub
git push -u origin main
```

**That's the ONLY manual upload!**

---

## 📊 Summary Table

| Destination | Upload Method | What You Do |
|------------|---------------|-------------|
| **GitHub** | ⚠️ **MANUAL** | Run `git push` commands |
| **HF Dataset** | ✅ **AUTOMATIC** | Run `python src/data_register.py` and `python src/data_prep.py` |
| **HF Model** | ✅ **AUTOMATIC** | Run `python src/train.py` |
| **HF Space** | ✅ **AUTOMATIC** | Run `python src/deploy_to_hf.py` |

---

## 🎯 Complete Workflow

### Step 1: Manual - GitHub (Do This First)
```bash
# Push everything to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Abhishektripathi/engine-predictive-maintenance.git
git push -u origin main
```
⏱️ **Time**: 2-3 minutes

---

### Step 2: Automatic - HF Dataset
```bash
export HF_TOKEN="your_hf_token"
python src/data_register.py
python src/data_prep.py
```
⏱️ **Time**: 1-2 minutes (scripts do everything)

---

### Step 3: Automatic - HF Model
```bash
python src/train.py
```
⏱️ **Time**: 5-10 minutes (trains model + auto-uploads)

---

### Step 4: Automatic - HF Space
```bash
python src/deploy_to_hf.py
```
⏱️ **Time**: 1-2 minutes (scripts do everything)

---

## ✅ What You Need to Prepare

### Before Running Scripts:

1. **Set Environment Variable:**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

2. **Update `src/config.py`** (already done):
   - HF repos are set to `Abhishektripathiak/...`

3. **That's it!** Scripts handle the rest automatically.

---

## 🔍 Verification

After running scripts, verify automatic uploads:

1. **HF Dataset**: https://huggingface.co/datasets/Abhishektripathiak/engine-maintenance-dataset
   - Should see: `data/engine_data.csv`, `data/train.csv`, `data/test.csv`

2. **HF Model**: https://huggingface.co/Abhishektripathiak/engine-maintenance-model
   - Should see: `model.joblib`

3. **HF Space**: https://huggingface.co/spaces/Abhishektripathiak/engine-maintenance-space
   - Should see: `src/app.py`, `Dockerfile`, `requirements.txt`
   - App should be running!

---

## 💡 Key Points

✅ **Automatic (3 out of 4):**
- HF Dataset upload
- HF Model upload  
- HF Space deployment

⚠️ **Manual (1 out of 4):**
- GitHub push (only this one!)

🎯 **Bottom Line:**
- **75% automatic** - Just run the scripts!
- **25% manual** - Just push to GitHub once!

---

## 🚨 Common Questions

**Q: Do I need to manually create HF repos?**
A: **No!** Scripts create them automatically if they don't exist.

**Q: Do I need to manually upload files to HF?**
A: **No!** Scripts upload everything automatically.

**Q: What if a script fails?**
A: Check `HF_TOKEN` is set, then re-run the script. It will continue from where it left off.

**Q: Can I skip GitHub and just use HF?**
A: For the project rubric, you need GitHub for the CI/CD workflow. But for HF-only, yes, scripts handle everything.

---

## 📝 Quick Checklist

- [ ] **Manual**: Push to GitHub (`git push`)
- [ ] **Automatic**: Run `python src/data_register.py`
- [ ] **Automatic**: Run `python src/data_prep.py`
- [ ] **Automatic**: Run `python src/train.py`
- [ ] **Automatic**: Run `python src/deploy_to_hf.py`
- [ ] Verify all uploads completed successfully

**Total Manual Work: ~5 minutes (just GitHub)**
**Total Automatic Work: ~10-15 minutes (scripts do everything)**
