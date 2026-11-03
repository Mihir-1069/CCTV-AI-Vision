const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const DEMO = (import.meta.env.VITE_USE_DEMO_DATA ?? "true") !== "false";

const demoCameras = [
  { id: "CAM-01", name: "Main Corridor", location: "Block A · Ground Floor", status: "online", people: 18, fps: 24, lastSeen: "Live" },
  { id: "CAM-02", name: "Classroom 204", location: "Block A · 2nd Floor", status: "online", people: 32, fps: 22, lastSeen: "Live" },
  { id: "CAM-03", name: "Library Entrance", location: "Block B · Ground Floor", status: "online", people: 9, fps: 25, lastSeen: "Live" },
  { id: "CAM-04", name: "Cafeteria", location: "Central Building", status: "offline", people: 0, fps: 0, lastSeen: "2 min ago" },
  { id: "CAM-05", name: "Science Lab", location: "Block C · 1st Floor", status: "online", people: 14, fps: 24, lastSeen: "Live" },
  { id: "CAM-06", name: "Sports Hall", location: "Sports Complex", status: "online", people: 21, fps: 23, lastSeen: "Live" }
];

const demoAlerts = [
  { id: "EV-1082", type: "Crowd density threshold", camera: "Main Corridor", time: "10:42:18 AM", severity: "medium", status: "Pending review", description: "People count stayed above the configured zone threshold." },
  { id: "EV-1081", type: "Camera stream unavailable", camera: "Cafeteria", time: "10:39:04 AM", severity: "high", status: "Open", description: "The camera stream stopped sending frames." },
  { id: "EV-1080", type: "Unusual movement pattern", camera: "Sports Hall", time: "10:31:52 AM", severity: "low", status: "Pending review", description: "Movement pattern flagged for human review." },
  { id: "EV-1079", type: "Zone occupancy limit", camera: "Classroom 204", time: "10:18:20 AM", severity: "medium", status: "Reviewed", description: "Zone occupancy exceeded the configured limit." }
];

async function request(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`API request failed: ${response.status}`);
  return response.json();
}

export async function getCameras() {
  if (DEMO) return demoCameras;
  return request("/api/cameras");
}

export async function getOccupancySummary() {
  if (DEMO) return { totalPeople: 94, activeCameras: 5, totalCameras: 6, alertsToday: 12, occupancyChange: "+8.2%" };
  return request("/api/occupancy/summary");
}

export async function getAlerts() {
  if (DEMO) return demoAlerts;
  return request("/api/alerts");
}

export async function getOccupancyAnalytics() {
  if (DEMO) return [
    { time: "08:00", count: 22 }, { time: "09:00", count: 48 },
    { time: "10:00", count: 76 }, { time: "11:00", count: 94 },
    { time: "12:00", count: 61 }, { time: "01:00", count: 38 },
    { time: "02:00", count: 82 }, { time: "03:00", count: 57 }
  ];
  return request("/api/analytics/occupancy");
}

export const isDemoMode = () => DEMO;
