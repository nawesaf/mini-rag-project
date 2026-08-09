# Mini RAG frontend

## Run locally

The development server proxies `/document` requests to FastAPI on `http://127.0.0.1:8000`.

```bash
# From the repository root
fastapi dev backend/main.py

# In a second terminal
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

For a separately hosted production build, copy `.env.example` to `.env.local` and set
`VITE_API_BASE_URL` to the backend origin. That deployment requires the backend to allow
the frontend origin with CORS, or both services to be exposed behind the same origin.
