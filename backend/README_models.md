Keeping model files in the repository
===================================

This project loads the following files at startup:

- `backend/vocab.pkl`
- `backend/models/encoder-3.pkl`
- `backend/models/decoder-3.pkl`

Important notes and recommended workflow
--------------------------------------

1) Use Git LFS for large binary files
   - GitHub and other Git hosts limit regular Git file sizes and don't handle large binary blobs efficiently.
   - Git LFS stores the actual binary content outside normal Git objects and keeps pointers in the repo, which keeps your repository fast and small.

2) Install and enable Git LFS locally

   On your machine (bash):

   ```bash
   # install (one-time)
   git lfs install

   # tell git-lfs to track model files (patterns already present in .gitattributes added at repo root)
   git lfs track "backend/models/*.pkl"
   git lfs track "backend/vocab.pkl"

   # add the .gitattributes (if it's not already added)
   git add .gitattributes

   # add model files and commit (if not already committed)
   git add backend/models/*.pkl backend/vocab.pkl
   git commit -m "Add model files via Git LFS"
   git push origin main
   ```

   If the model files were already committed to Git (not LFS), migrating them into LFS requires history rewrite:

   ```bash
   # rewrites history to move matching files to LFS (use carefully, especially on shared branches)
   git lfs migrate import --include="backend/models/*.pkl,backend/vocab.pkl"
   git push --force origin main
   ```

   CAUTION: `git lfs migrate` rewrites history and requires force-push. Coordinate with collaborators before using it.

3) Repository size and hosting limits
   - Git LFS reduces Git repo size but your Git host (GitHub/GitLab) may have LFS bandwidth/storage limits on free plans. Check your host's LFS quotas.
   - GitHub has a 100 MB per-file hard limit for normal Git objects; Git LFS bypasses this.

4) Render deployment notes
   - Render clones your repository at build time; if model files are in the repo (LFS or not), they will be present in the build container and available to the running service.
   - Large repo + large models increase build time and may increase build memory/disk usage. If builds fail due to resource limits, consider hosting models externally (S3) and downloading at startup.
   - The runtime filesystem on Render is ephemeral across deploys. Storing the model in the repo (and thus in the deployed container) is fine for serving; if you need persistent writable storage between deploys, use an external store.

5) Production memory consideration
   - Your app loads PyTorch models at startup. Avoid multiple Gunicorn workers to prevent duplicated memory usage. The included `Procfile` uses a single worker and threads.

Alternate: Keep models in S3 or other object storage if you:
   - need to update them frequently without pushing code
   - want to reduce repo size and build time
   - require private access control separate from the repo

If you'd like, I can:
- add a helper that downloads models from S3 at startup when the files are missing, or
- add a CI check to ensure models are present before deployment.
