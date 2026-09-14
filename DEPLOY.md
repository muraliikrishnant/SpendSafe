# Deploying SpendSafe

Frontend on **GitHub Pages** (static, free), backend on **Render** (free tier).
The browser never computes an affordability answer — it asks the Python engine
in `code/`, so the web app and the batch pipeline can't drift apart on the math.

```
docs/index.html   →  GitHub Pages   (https://<you>.github.io/SpendSafe/)
server/           →  Render         (https://spendsafe-api.onrender.com)
   └── imports code/safety_engine.py, code/spending_changes.py unmodified
```

---

## 1. Backend on Render

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the `SpendSafe` repo. Render reads [`render.yaml`](./render.yaml) and
   proposes a service called **spendsafe-api**. Click **Apply**.
4. Wait for the first build (~2 min). Check it's alive:

   ```bash
   curl https://spendsafe-api.onrender.com/health
   # {"status":"ok","horizon_days":90}
   ```

### Environment variables

Set these in the Render dashboard under the service → **Environment**:

| Key | Value | Needed? |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://<your-github-username>.github.io` | Yes |
| `NVIDIA_API_KEY` | your build.nvidia.com key | Optional |
| `NVIDIA_TEXT_MODEL` | `openai/gpt-oss-20b` | Optional |

`NVIDIA_API_KEY` only powers plain-English parsing ("a 40k phone next month").
Leave it unset and the page reads the number out of your sentence itself —
every financial calculation works either way, because no model is involved in
them. **Never put this key in `docs/`** — it would be public.

---

## 2. Frontend on GitHub Pages

1. In [`docs/index.html`](./docs/index.html), check the `API` constant near the
   top of the `<script>` block points at your Render URL:

   ```js
   const API = window.location.hostname.endsWith("github.io")
     ? "https://spendsafe-api.onrender.com"   // ← yours, if the name differs
     : "http://localhost:8811";
   ```

2. On GitHub: **Settings** → **Pages** → Source: **Deploy from a branch**,
   Branch: **main**, Folder: **/docs** → **Save**.
3. Wait ~1 minute. It's live at `https://<your-username>.github.io/SpendSafe/`.

---

## Running it locally

```bash
# backend
source venv/bin/activate
pip install -r server/requirements.txt
cd server && uvicorn main:app --port 8811 --reload

# frontend (separate terminal) — any static server works
cd docs && python3 -m http.server 8000
# open http://localhost:8000
```

The frontend auto-detects localhost and talks to `http://localhost:8811`.

---

## Things worth knowing

- **The free Render instance sleeps after ~15 minutes idle.** The first request
  after that takes 30–60 seconds while it wakes. The page shows an "engine
  waking" state and tells the user to retry rather than looking broken. A paid
  instance ($7/mo) removes this.
- **No financial data is stored server-side.** A request carries the numbers,
  the response carries the answer, nothing is written to disk. Your profile
  lives in your own browser's `localStorage`.
- **This is decision-support, not financial advice**, and it's fed by numbers
  you type in by hand. Wiring it to real bank data (Plaid, TrueLayer, Account
  Aggregator) needs a registered entity and a compliance review — the data layer
  is deliberately swappable (`server/engine_adapter.py::build_ledger`) so that
  change doesn't touch the engine.
