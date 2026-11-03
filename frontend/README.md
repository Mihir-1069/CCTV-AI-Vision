# CCTV AI Vision — Frontend

React + Vite frontend prototype for the CCTV AI Vision project.

## Features
- Overview dashboard with camera status and occupancy
- Live Monitoring page (demo stream cards)
- Camera Management
- Incident Center with review actions
- Analytics charts
- Settings page
- Responsive dark UI
- Demo data by default; API adapter is in `src/services/api.js`

## Run
```bash
npm install
npm run dev
```
Open the local URL printed by Vite (usually http://localhost:5173).

## Connect FastAPI
1. Copy `.env.example` to `.env`.
2. Set `VITE_USE_DEMO_DATA=false`.
3. Set `VITE_API_BASE_URL` to your backend URL.
4. Adjust endpoint paths and response shapes in `src/services/api.js` to match your backend.

The UI currently expects:
- `GET /api/cameras`
- `GET /api/occupancy/summary`
- `GET /api/alerts`
- `GET /api/analytics/occupancy`

The backend is not included in this frontend package. Demo mode is enabled by default.
