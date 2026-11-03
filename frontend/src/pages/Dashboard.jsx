import { useEffect, useState } from "react";
import { Activity, Camera, CircleAlert, Users, ArrowRight, Clock3 } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import StatCard from "../components/StatCard";
import CameraCard from "../components/CameraCard";
import AlertCard from "../components/AlertCard";
import { getCameras, getOccupancySummary, getAlerts, getOccupancyAnalytics } from "../services/api";

export default function Dashboard({ navigate }) {
  const [cameras, setCameras] = useState([]);
  const [summary, setSummary] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [chartData, setChartData] = useState([]);
  useEffect(() => {
    Promise.all([getCameras(), getOccupancySummary(), getAlerts(), getOccupancyAnalytics()])
      .then(([c, s, a, chart]) => { setCameras(c); setSummary(s); setAlerts(a); setChartData(chart); })
      .catch((e) => console.error("Dashboard load failed", e));
  }, []);

  return <div className="page-content">
    <div className="welcome-row"><div><div className="section-kicker">MONDAY, OCTOBER 31, 2026</div><h2>Good morning, Admin <span>✦</span></h2><p>Here’s what’s happening across your campus today.</p></div><div className="demo-pill"><i /> Demo data · Preview mode</div></div>
    <div className="stats-grid">
      <StatCard label="People on campus" value={summary.totalPeople ?? "—"} change={summary.occupancyChange} note="vs. previous hour" icon={Users} tone="blue" />
      <StatCard label="Active cameras" value={`${summary.activeCameras ?? "—"}/${summary.totalCameras ?? "—"}`} note="Camera connections" icon={Camera} tone="green" />
      <StatCard label="Alerts today" value={summary.alertsToday ?? "—"} change="2 new" note="Requires review" icon={CircleAlert} tone="orange" />
      <StatCard label="System uptime" value="99.8%" note="Last 30 days" icon={Activity} tone="purple" />
    </div>

    <div className="content-grid">
      <section className="panel occupancy-panel">
        <div className="panel-heading"><div><h3>Campus occupancy</h3><p>People detected across monitored zones</p></div><button className="period-select">Today <span>⌄</span></button></div>
        <div className="chart-summary"><strong>{summary.totalPeople ?? 94}</strong><span className="chart-unit">people currently detected</span><span className="chart-change">↑ 8.2%</span></div>
        <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
          <defs><linearGradient id="occupancyGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5c8cff" stopOpacity={0.28}/><stop offset="100%" stopColor="#5c8cff" stopOpacity={0.01}/></linearGradient></defs>
          <CartesianGrid stroke="#202a3c" strokeDasharray="3 5" vertical={false}/>
          <XAxis dataKey="time" tick={{ fill: "#7f8ba3", fontSize: 11 }} axisLine={false} tickLine={false}/>
          <YAxis tick={{ fill: "#7f8ba3", fontSize: 11 }} axisLine={false} tickLine={false}/>
          <Tooltip contentStyle={{ background: "#111a2b", border: "1px solid #2a3850", borderRadius: 10, color: "#edf3ff" }} />
          <Area type="monotone" dataKey="count" stroke="#6b96ff" strokeWidth={2.5} fill="url(#occupancyGradient)" />
        </AreaChart></ResponsiveContainer></div>
      </section>
      <section className="panel alerts-panel">
        <div className="panel-heading"><div><h3>Recent alerts <span className="count-badge">{alerts.length}</span></h3><p>Events that may need attention</p></div><button className="link-button" onClick={() => navigate("incidents")}>View all <ArrowRight size={14}/></button></div>
        <div className="alert-list">{alerts.slice(0, 4).map(a => <AlertCard key={a.id} alert={a} compact />)}</div>
      </section>
    </div>

    <section className="panel cameras-panel">
      <div className="panel-heading"><div><h3>Camera overview</h3><p>Live status of connected cameras</p></div><button className="link-button" onClick={() => navigate("cameras")}>Manage cameras <ArrowRight size={14}/></button></div>
      <div className="camera-grid">{cameras.slice(0, 3).map(camera => <CameraCard key={camera.id} camera={camera} />)}</div>
    </section>
    <div className="footer-note"><Clock3 size={13}/> Last refreshed just now <span>·</span> This interface uses sample data for demonstration.</div>
  </div>;
}