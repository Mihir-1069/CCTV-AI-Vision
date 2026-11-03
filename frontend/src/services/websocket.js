// Optional WebSocket helper. Adapt the event schema to your FastAPI backend.
export function connectAlerts(onMessage) {
  const base = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
  const wsBase = base.replace(/^http/, "ws");
  const socket = new WebSocket(`${wsBase}/ws/alerts`);

  socket.onmessage = (event) => {
    try { onMessage(JSON.parse(event.data)); }
    catch (error) { console.error("Invalid WebSocket message", error); }
  };

  return socket;
}