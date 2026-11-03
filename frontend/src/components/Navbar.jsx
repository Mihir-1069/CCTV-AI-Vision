import { Bell, Search, Command, ChevronDown } from "lucide-react";

const titles = {
  dashboard: ["Overview", "School-wide monitoring at a glance"],
  live: ["Live monitoring", "View camera feeds and real-time activity"],
  cameras: ["Camera management", "Monitor and configure connected cameras"],
  incidents: ["Incident center", "Review AI-generated event candidates"],
  analytics: ["Analytics", "Explore occupancy and event trends"],
  settings: ["Settings", "Configure your monitoring workspace"]
};

export default function Navbar({ page }) {
  const [title, subtitle] = titles[page] || titles.dashboard;
  return <header className="topbar">
    <div className="page-heading"><div className="eyebrow">CCTV AI VISION / WORKSPACE</div><h1>{title}</h1><p>{subtitle}</p></div>
    <div className="top-actions">
      <button className="search-button"><Search size={16} /><span>Search</span><kbd><Command size={11} /> K</kbd></button>
      <button className="icon-button notification-button" aria-label="Notifications"><Bell size={18} /><i /></button>
      <div className="top-user"><div className="avatar small">AD</div><ChevronDown size={15} /></div>
    </div>
  </header>;
}