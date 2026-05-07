# SpamShield Frontend

React + Vite + Tailwind CSS frontend for the SMS Spam Detector API.

## Stack
- React 18
- React Router DOM v6
- Tailwind CSS v3
- Vite

## Setup

```bash
npm install
npm run dev
```

App runs at `http://localhost:5173` and proxies `/api/*` → `http://localhost:8000`.

Make sure your FastAPI backend is running on port 8000 before starting.

## Pages
- `/` — Single message prediction
- `/batch` — Multi-message batch scan
- `/models` — Model comparison & metrics
- `/explain` — AI explainability (Groq LLaMA 3.3 70B)
