# Workspace layout (RunPod / persistent volume)

This folder mirrors the layout used on the RunPod pod under `/workspace/`:

```
workspace/
├── jaguar-reid-kaggle/     # Project folder
│   ├── data/               # Datasets (raw, processed, train_crops, etc.)
│   ├── scripts/            # Python scripts
│   ├── requirements.txt    # Dependencies
│   └── .cursorrules        # Cursor AI instructions (optional)
└── venv/                   # Virtual environment (persistent)
```

- **`venv/`** – Python virtual environment. Create and use it with:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r jaguar-reid-kaggle/requirements.txt
  ```
- **`jaguar-reid-kaggle/`** – Project code, data, and config.

On the pod, after cloning the repo into `/workspace/`, run from `/workspace/`:
```bash
cd /workspace
source venv/bin/activate
cd jaguar-reid-kaggle
```
---

## Data and RunPod: where to store images

- **Do images need to be on the pod?** Yes. When you run scripts (e.g. `crop_jaguars_sam3.py`), the code reads images from disk, so they must be available on the pod—either on the pod’s filesystem or on a **mounted volume**. They do not need to live in the repo.

- **Do not push images to GitHub.** The repo has `data/` in `.gitignore`. Keep only code and small config/manifests in Git; store the actual image data elsewhere.

**Ways to get images onto RunPod:**

1. **RunPod Network Volume** (recommended for large, reusable data)  
   Create a network volume in the RunPod dashboard, upload your jaguar images there once (e.g. via SFTP/rsync). Attach the volume to your pod so the data is available at a path like `/workspace/jaguar-reid-kaggle/data/...`. No need to re-upload when you start a new pod.

2. **Download on pod startup**  
   If the dataset is on **Kaggle**, use the Kaggle API on the pod (e.g. in a startup script):
   ```bash
   pip install kaggle
   # Configure ~/.kaggle/kaggle.json with your API key
   kaggle datasets download -d <your-dataset> -p jaguar-reid-kaggle/data/raw_gallery/
   unzip ...
   ```
   If data is in **S3/GCS**, use `rclone` or `aws s3 sync` in a startup script to pull into `data/`.

3. **One-time copy from your machine**  
   When the pod is running, use `rsync` after `scripts/update_ssh.py`:
   ```bash
   rsync -avz --progress /path/to/local/jaguars_images/ runpod-spot:/workspace/jaguar-reid-kaggle/data/raw_gallery/jaguars_images/
   ```

In all cases, keep the same paths in your scripts (`data/raw_gallery/jaguars_images`, etc.); only where the data is stored (volume vs. download vs. rsync) changes.

---

## First-time pod setup (order of operations)

The repo does **not** contain images. Cloning only gives you code. Do this in order:

1. **Start the pod** (RunPod dashboard) with your (resized) volume attached. If the pod was already running, stop it and start it again so it picks up the new volume size. Wait until the pod status is "Running".

2. **On the pod** (SSH or RunPod's web terminal), from `/workspace`:
   ```bash
   cd /workspace
   git clone <your-repo-url> temp-clone
   mv temp-clone/jaguar-reid-kaggle ./
   rm -rf temp-clone
   python3 -m venv venv
   source venv/bin/activate
   pip install -r jaguar-reid-kaggle/requirements.txt
   mkdir -p jaguar-reid-kaggle/data/raw_gallery/jaguars_images
   mkdir -p jaguar-reid-kaggle/data/train_crops
   mkdir -p jaguar-reid-kaggle/data/processed_gallery
   ```
   That gives you the code and the folder structure; the data dirs are still empty.

3. **Sync the images from your Mac** (run this on your Mac, not on the pod). First run `python workspace/jaguar-reid-kaggle/scripts/update_ssh.py` so `runpod-spot` in `~/.ssh/config` points at the pod. Then:
   ```bash
   # Raw gallery (~108 MB)
   rsync -avz --progress workspace/jaguar-reid-kaggle/data/raw_gallery/ runpod-spot:/workspace/jaguar-reid-kaggle/data/raw_gallery/

   # Train crops (~14 GB) — only if you need them on the pod
   rsync -avz --progress workspace/jaguar-reid-kaggle/data/train_crops/ runpod-spot:/workspace/jaguar-reid-kaggle/data/train_crops/
   ```
   After this, the images are where the scripts expect them (`data/raw_gallery/...`, `data/train_crops/...`).

**Summary:** Clone first (code + empty data dirs), then sync images. You don't sync "into the repo before cloning"—you clone on the pod, then fill the data dirs by rsync (or by downloading from Kaggle/S3 on the pod). If `/workspace` is on your RunPod volume, everything you put in `/workspace/jaguar-reid-kaggle/data/` persists for the next time you start a pod with that volume attached.
