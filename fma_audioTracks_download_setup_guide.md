# EMPATHY: Downloading the ~2,000 FMA Audio Tracks on the Azure VM

Downloads ~2,000 full-length, Echonest-annotated tracks from `fma_full` onto the Azure VM without storing the full 879 GiB archive (Path A), or with a temporary 1 TB disk if the host does not support partial downloads (Path B), followed by audio verification, loudness normalization, and a FastAPI/Nginx smoke test.

## Links

| Resource | URL |
| --- | --- |
| FMA dataset repo (README, usage notebooks, checksums) | https://github.com/mdeff/fma |
| `fma_full.zip` (untrimmed audio, 106,574 tracks, ~879 GiB) | https://os.unil.cloud.switch.ch/fma/fma_full.zip |
| `fma_metadata.zip` (342 MiB: `tracks.csv`, `echonest.csv`, `features.csv`, `genres.csv`) | https://os.unil.cloud.switch.ch/fma/fma_metadata.zip |
| Partial-unzip tool: `python-remotezip` | https://github.com/gtsystem/python-remotezip |
| Azure portal | https://portal.azure.com |
| Azure CLI install (only needed outside Cloud Shell) | https://learn.microsoft.com/cli/azure/install-azure-cli |
| Azure pricing calculator | https://azure.microsoft.com/pricing/calculator/ |

Checksums (from the FMA README; confirm against https://github.com/mdeff/fma before relying on them):

```
f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip
0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  fma_full.zip
```

## Prerequisites

- Azure for Students subscription active ($100 credit) and a budget alert set (Cost Management + Billing → Budgets).
- Resource group `rg-empathy` and VM `vm-empathy` (Ubuntu 22.04, `Standard_B2als_v2` or `Standard_B2s`, 64 GB OS disk) created, with inbound TCP port `22` (SSH) and `80` (HTTP) allowed in Network Security Group rules.
- SSH access: `ssh azureuser@<VM_PUBLIC_IP>`.

## Disk budget (64 GB OS disk)

| Item | Approx. size |
| --- | --- |
| `fma_metadata.zip` + extracted CSVs | ~1.5 GB |
| Raw selected tracks (`/srv/audio_raw`) | ~15–25 GB |
| Normalized tracks (`/srv/audio`) | ~11 GB |

Check space with `df -h /`. Delete `/srv/audio_raw` after normalization is verified. If space is short, attach a data disk (see Path B for the commands) or reduce the pool to ~1,000 tracks.

## Step 1: Connect and install dependencies

```bash
ssh azureuser@<VM_PUBLIC_IP>

sudo apt update && sudo apt install -y python3-venv python3-pip unzip git ffmpeg tmux curl nginx
sudo mkdir -p /srv/audio_raw /srv/audio && sudo chown -R azureuser:azureuser /srv/audio_raw /srv/audio

python3 -m venv ~/venv && source ~/venv/bin/activate
pip install remotezip pandas pyarrow requests ffmpeg-normalize fastapi uvicorn
mkdir -p ~/data && cd ~/data
```

## Step 2: Download the metadata (small) and verify

```bash
cd ~/data
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
unzip fma_metadata.zip
ls fma_metadata
```

Expected: `tracks.csv`, `echonest.csv`, `features.csv`, `genres.csv`, `raw_*.csv`.

Echonest descriptors exist for **13,129 tracks only**. FMA audio supplies the playable files; Echonest supplies valence, energy, acousticness, tempo, etc.

## Step 3: Build the ~2,000-track selection (`pool_selected.csv`)

Save as `~/data/build_pool.py`:

```python
import pandas as pd

echo = pd.read_csv("fma_metadata/echonest.csv", index_col=0, header=[0, 1, 2])
af = echo["echonest"]["audio_features"].copy()  # acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence

tr = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0, 1])
af["duration"]  = tr[("track", "duration")]
af["license"]   = tr[("track", "license")]
af["genre_top"] = tr[("track", "genre_top")]
af["title"]     = tr[("track", "title")]
af["artist"]    = tr[("artist", "name")]

pool = af.dropna(subset=["energy", "valence"])
pool = pool[pool.duration.between(120, 360)]                              # full songs, 2-6 min
pool = pool[pool.license.astype(str).str.contains("Creative Commons")]    # read each license for your use

# Balanced coverage of the energy x valence quadrants
pool["quad"] = (pool.energy > pool.energy.median()).astype(int) * 2 + \
               (pool.valence > pool.valence.median()).astype(int)
sel = pool.groupby("quad", group_keys=False).apply(
    lambda d: d.sample(min(len(d), 500), random_state=42))

sel.to_csv("pool_selected.csv")
print(len(sel), "tracks selected")
print(sel.quad.value_counts().sort_index())
```

```bash
cd ~/data && python build_pool.py
```

Expected: up to 2,000 rows (500 per quadrant, fewer if a quadrant has less after filtering). The index of `pool_selected.csv` holds the FMA track IDs used in the next steps.

## Step 4: Test whether the host supports partial downloads

```bash
curl -sI https://os.unil.cloud.switch.ch/fma/fma_full.zip | grep -i -E "accept-ranges|content-length"
```

- `Accept-Ranges: bytes` present → **Path A** (recommended; no 879 GiB disk needed).
- Absent → **Path B**.

## Step 5, Path A: Partial extraction with `remotezip`

`remotezip` reads the zip's central directory over HTTP range requests and downloads only the requested files. Repo: https://github.com/gtsystem/python-remotezip

Save as `~/data/download_tracks.py`:

```python
import os, time
import pandas as pd
from remotezip import RemoteZip

URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
OUT = "/srv/audio_raw"
ids = pd.read_csv("pool_selected.csv", index_col=0).index

def member_name(tid):
    s = f"{tid:06d}"
    return f"fma_full/{s[:3]}/{s}.mp3"

with RemoteZip(URL) as z:
    names = set(z.namelist())
    print("Sample member names:", list(names)[:3])   # confirm the prefix; edit member_name() if different

missing, failed = [], []
for i, tid in enumerate(ids, 1):
    n = member_name(tid)
    dest = os.path.join(OUT, n)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        continue                                      # resume support
    if n not in names:
        missing.append(tid); continue
    for attempt in range(4):
        try:
            with RemoteZip(URL) as z:                 # fresh connection per file is more robust
                z.extract(n, OUT)
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    else:
        failed.append(tid)
    if i % 100 == 0:
        print(f"{i}/{len(ids)} processed", flush=True)

pd.Series(missing).to_csv("missing_in_zip.csv", index=False)
pd.Series(failed).to_csv("failed_downloads.csv", index=False)
print("Done. Missing in zip:", len(missing), "| Failed:", len(failed))
```

Run inside `tmux` so it survives SSH disconnects:

```bash
tmux new -s dl
source ~/venv/bin/activate && cd ~/data
python download_tracks.py
# detach: Ctrl+b then d    | reattach: tmux attach -t dl
```

Re-run the script to retry anything in `failed_downloads.csv` (existing files are skipped). Tracks listed in `missing_in_zip.csv` are not in the archive; drop them from the pool or resample.

If `RemoteZip` fails or hangs on the central directory (the archive is very large), switch to Path B.

## Step 5, Path B: Full archive on a temporary data disk

Use only if Step 4 shows no range support. A 1 TB disk is needed because the zip is ~879 GiB.

```bash
# On your laptop or Cloud Shell
az vm disk attach -g rg-empathy --vm-name vm-empathy --name data1 --new --size-gb 1024 --sku Standard_LRS
```

```bash
# On the VM
lsblk                                   # find the new disk, e.g. /dev/sdc
sudo parted /dev/sdc --script mklabel gpt mkpart xfspart xfs 0% 100%
sudo mkfs.xfs /dev/sdc1
sudo mkdir /datadrive && sudo mount /dev/sdc1 /datadrive && sudo chown azureuser /datadrive

cd /datadrive && tmux new -s dl
curl -C - -O https://os.unil.cloud.switch.ch/fma/fma_full.zip      # -C - resumes interrupted downloads
echo "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  fma_full.zip" | sha1sum -c -
```

Extract only the selected tracks:

```python
import zipfile, pandas as pd
ids = pd.read_csv("/home/azureuser/data/pool_selected.csv", index_col=0).index
with zipfile.ZipFile("/datadrive/fma_full.zip") as z:
    names = set(z.namelist())
    for tid in ids:
        s = f"{tid:06d}"; n = f"fma_full/{s[:3]}/{s}.mp3"
        if n in names:
            z.extract(n, "/srv/audio_raw")
```

Stop billing immediately afterwards:

```bash
rm /datadrive/fma_full.zip
sudo umount /datadrive
# On your laptop or Cloud Shell
az vm disk detach -g rg-empathy --vm-name vm-empathy -n data1
az disk delete -g rg-empathy -n data1 --yes
```

Inbound traffic to Azure is free; the disk is billed prorated, so keep it only as long as needed.

## Step 6: Verify the download and filter the metadata CSV

```bash
find /srv/audio_raw -name "*.mp3" | wc -l            # should be close to 2000
du -sh /srv/audio_raw
```

Check that every selected ID exists, decodes cleanly with `ffprobe`, and write the verified track list (`pool_final.csv`):

```python
import os, subprocess, pandas as pd

df = pd.read_csv("/home/azureuser/data/pool_selected.csv", index_col=0)
have, bad = [], []
for tid in df.index:
    s = f"{tid:06d}"
    p = f"/srv/audio_raw/fma_full/{s[:3]}/{s}.mp3"
    if not os.path.exists(p):
        continue
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
        capture_output=True, text=True
    )
    (have if r.returncode == 0 and r.stdout.strip() else bad).append(tid)

print("OK:", len(have), "| Corrupt/unreadable:", len(bad), "| Not downloaded:", len(df) - len(have) - len(bad))
pd.Series(have).to_csv("/home/azureuser/data/pool_final_ids.csv", index=False, header=["track_id"])
df.loc[have].to_csv("/home/azureuser/data/pool_final.csv")
```

Re-download any corrupt IDs (delete the corrupt file, then re-run Step 5). Use `pool_final.csv` as the canonical metadata table for the backend app.

## Step 7: Normalize loudness in parallel (recommended)

Loudness differences between tracks confound the `volumechange` signal. Run normalization inside `tmux` using parallel workers (`ProcessPoolExecutor`) while preserving the 3-digit subfolder layout (`/srv/audio/<subdir>/<track_id>.mp3`) and skipping files that already exist.

Save as `~/data/normalize_audio.py`:

```python
import os, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_ROOT = "/srv/audio_raw/fma_full"
OUT_ROOT = "/srv/audio"

tasks = []
for root, _, files in os.walk(RAW_ROOT):
    for f in files:
        if not f.endswith(".mp3"):
            continue
        src = os.path.join(root, f)
        rel = os.path.relpath(src, RAW_ROOT)      # e.g. 014/014391.mp3
        dst = os.path.join(OUT_ROOT, rel)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            tasks.append((src, dst))

def normalize_one(pair):
    src, dst = pair
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        "ffmpeg-normalize", src,
        "-nt", "ebu", "-t", "-16",
        "-c:a", "libmp3lame", "-b:a", "192k",
        "-o", dst, "-f"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return src, res.returncode, res.stderr

print(f"Tracks remaining to normalize: {len(tasks)}", flush=True)
failed = []
with ProcessPoolExecutor(max_workers=2) as ex:
    futures = [ex.submit(normalize_one, t) for t in tasks]
    for i, fut in enumerate(as_completed(futures), 1):
        src, rc, err = fut.result()
        if rc != 0:
            failed.append((src, err))
        if i % 100 == 0 or i == len(tasks):
            print(f"{i}/{len(tasks)} normalized (failed: {len(failed)})", flush=True)
```

Run inside `tmux` so SSH disconnects do not interrupt processing:

```bash
tmux new -s norm
source ~/venv/bin/activate && cd ~/data
python normalize_audio.py
# detach: Ctrl+b then d    | reattach: tmux attach -t norm
```

Verify all normalized tracks are present before deleting raw files:

```bash
find /srv/audio -name "*.mp3" | wc -l && du -sh /srv/audio
```

Expected: `2000` (matching `pool_final.csv`) and `~11G` disk usage. Output paths follow `/srv/audio/<subdir>/<6-digit-track-id>.mp3` (e.g., `/srv/audio/014/014391.mp3`).

## Step 8: Smoke-test FastAPI + Nginx audio streaming

Verify that the VM serves the normalized MP3 files over HTTP (including byte-range seeking) before removing raw files or deallocating the VM.

1. **Create a minimal `~/data/smoke_test.py`:**

```python
import os, random
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
AUDIO_DIR = os.getenv("AUDIO_DIR", "/srv/audio")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

@app.get("/health")
def health():
    files = [os.path.join(r, f) for r, _, fs in os.walk(AUDIO_DIR) for f in fs if f.endswith(".mp3")]
    sample = os.path.relpath(random.choice(files), AUDIO_DIR).replace("\\", "/") if files else None
    return {"status": "ok", "track_count": len(files), "sample_url": f"/audio/{sample}" if sample else None}
```

2. **Start Uvicorn and configure Nginx as a reverse proxy on port 80:**

```bash
nohup ~/venv/bin/uvicorn smoke_test:app --app-dir ~/data --host 127.0.0.1 --port 8000 > ~/uvicorn.log 2>&1 &

sudo tee /etc/nginx/sites-available/empathy > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/empathy /etc/nginx/sites-enabled/empathy
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

3. **Verify in a browser:**
   - Open `http://<VM_PUBLIC_IP>/health` → confirms `"status": "ok"`, `"track_count": 2000`, and a `"sample_url"` path.
   - Open `http://<VM_PUBLIC_IP><sample_url>` (e.g., `http://<VM_PUBLIC_IP>/audio/014/014391.mp3`) → confirms the normalized track plays cleanly in the browser.

## Step 9: Clean up, sync local copy, and save credit

Once Step 7 and Step 8 are verified, remove `/srv/audio_raw` (`sudo` is required because `/srv` is owned by `root`) and the raw metadata zip:

```bash
sudo rm -rf /srv/audio_raw && df -h /
rm -f ~/data/fma_metadata.zip
```

- **Optional local sync for offline development:** From your local machine, download `pool_final.csv` and `/srv/audio` via `scp` (`scp -r -i <key.pem> azureuser@<VM_PUBLIC_IP>:/srv/audio ./audio`) and point your local backend to `./audio` via `AUDIO_DIR`.
- **Stop compute billing when idle:** Run `az vm deallocate -g rg-empathy -n vm-empathy` (or click **Stop** in Azure Portal → **Overview** until status shows **Stopped (deallocated)**). Restart later with `az vm start -g rg-empathy -n vm-empathy`. Note that managed disks still incur minor storage charges while deallocated.
- **Version control:** Back up `pool_selected.csv`, `pool_final.csv`, and the Python scripts to your GitHub repo. **Do not commit audio files or participant logs.**
- **End of study:** Delete all Azure resources with `az group delete -n rg-empathy --yes`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `RequestDisallowedByAzure` / size error when creating the VM | Retry in another region (`eastasia`, `japaneast`) or list sizes: `az vm list-skus -l southeastasia --size Standard_B -o table` |
| `namelist()` shows a different path prefix | Edit `member_name()` / the extraction pattern to match |
| Download stalls or drops | Re-run `download_tracks.py` (skips finished files); use `curl -C -` for Path B |
| SSH disconnect killed the job | Always run inside `tmux`; reattach with `tmux attach -t dl` or `tmux attach -t norm` |
| `rm: cannot remove '/srv/audio_raw': Permission denied` | Use `sudo rm -rf /srv/audio_raw` because `/srv` is owned by `root` |
| `No space left on device` | `df -h /`; delete `/srv/audio_raw` after normalization, or attach a data disk |
| Fewer than ~2,000 tracks after filtering | Raise the per-quadrant sample cap, widen the duration filter, or accept a smaller pool (~1,000 is sufficient) |
| Missing IDs in the archive | Drop them from the pool (`missing_in_zip.csv`) and use `pool_final.csv` |

## Citation and licensing

Cite the dataset: Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). *FMA: A dataset for music analysis.* https://arxiv.org/abs/1612.01840 (verify the APA entry). Cite `python-remotezip`, `ffmpeg-normalize`, and any other reused tools. Check each track's Creative Commons license for your use case and keep attribution information from `tracks.csv`.

*Note: This .md file was entirely generated by Claude and Gemini, but was verified by humans.*
