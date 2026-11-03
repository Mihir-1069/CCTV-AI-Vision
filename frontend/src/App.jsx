import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import LiveMonitoring from "./pages/LiveMonitoring";
import CameraManagement from "./pages/CameraManagement";
import IncidentCenter from "./pages/IncidentCenter";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";
import { isDemoMode } from "./services/api";

export default function App() {
  const [page, setPage] = useState("dashboard");
  const content = {
    dashboard: <Dashboard navigate={setPage}/>,
    live: <LiveMonitoring/>,
    cameras: <CameraManagement/>,
    incidents: <IncidentCenter/>,
    analytics: <Analytics/>,
    settings: <Settings/>
  };
  return <div className="app-shell">
    <Sidebar active={page} onNavigate={setPage}/>
    <main className="main-area">
      <Navbar page={page}/>
      {isDemoMode() && <div className="demo-notice"><span className="demo-notice-dot"/> Demo mode is active. Data shown is illustrative, not from real school cameras.</div>}
      {content[page] || content.dashboard}
    </main>
  </div>;
}