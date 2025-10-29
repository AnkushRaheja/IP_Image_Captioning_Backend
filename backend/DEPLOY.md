Render deployment steps (quick)

1. Confirm .pkl files are tracked

   - I removed the `*.pkl` and `models/*.pkl` ignore rules from `.gitignore` so `vocab.pkl` and `models/*.pkl` will be committed.

2. Commit and push (from repo root or using the backend folder):

```bash
# from repo root (recommended)
cd "d:/Projects/ML project/IP_Image_Captioning_Backend"
# stage model files and changes
git add backend/.gitignore backend/vocab.pkl backend/models/encoder-3.pkl backend/models/decoder-3.pkl backend/Procfile backend/DEPLOY.md && git add -A
# commit
git commit -m "Include model .pkl files and prepare for Render deployment"
# push
git push origin main
```

Notes:

- GitHub rejects files larger than 100 MB. If any .pkl file is >100 MB, pushing will fail and you must use Git LFS or host the model externally (S3, Google Cloud Storage). If you expect large model files, tell me and I can switch the repo to use Git LFS and update the deploy flow.

Render setup (manual via Render dashboard)

- Create a new Web Service and connect your GitHub repo.
- Important: set the "Root Directory" (or "Service Directory") to `backend` so Render builds and serves from the backend folder.
- Build command: leave default (Render will run pip install -r requirements.txt from the root directory). If asked, use: `pip install -r requirements.txt`.
- Start command: Render will use the `Procfile` in `backend` which contains:
  `web: gunicorn -w 1 -k gthread --threads 8 --bind 0.0.0.0:$PORT app:app`
- Environment variables: you can optionally set `VOCAB_PATH`, `ENCODER_PATH`, and `DECODER_PATH` if you move model files.

Quick health check after deploy

- After deploy, curl the health endpoint:

```bash
curl https://<your-render-service>.onrender.com/health
```

If the response says `status: healthy` and shows `vocabulary_size`, you're good.
