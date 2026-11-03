import { Activity, Camera, ChartNoAxesCombined, CircleAlert, LayoutDashboard, Settings, ShieldCheck, Video } from "lucide-react";

const navItems = [
  { id: "dashboard", label: "Overview", icon: LayoutDashboard },
  { id: "live", label: "Live monitoring", icon: Video },
  { id: "cameras", label: "Camera management", icon: Camera },
  { id: "incidents", label: "Incident center", icon: CircleAlert, count: "4" },
  { id: "analytics", label: "Analytics", icon: ChartNoAxesCombined },
  { id: "settings", label: "Settings", icon: Settings }
];

export function Sidebar({ active, onNavigate }) {
  return <aside className="sidebar">
    <div className="brand">
      <div className="brand-mark"><ShieldCheck size={21} /></div>
      <div><div className="brand-name">CCTV AI <span>Vision</span></div><div className="brand-caption">INTELLIGENCE PLATFORM</div></div>
    </div>
    <div className="workspace-label">WORKSPACE</div>
    <nav className="nav-list">
      {navItems.map(({ id, label, icon: Icon, count }) => <button key={id} className={`nav-item ${active === id ? "active" : ""}`} onClick={() => onNavigate(id)}>
        <Icon size={18} strokeWidth={1.8} /><span>{label}</span>{count && <b className="nav-count">{count}</b>}
      </button>)}
    </nav>
    <div className="sidebar-bottom">
      <div className="system-card">
        <div className="system-top"><span className="pulse-dot" /> System status</div>
        <strong>All systems operational</strong>
        <div className="system-meter"><i /></div>
        <small>5 of 6 cameras connected</small>
      </div>
      <div className="profile">
        <div className="avatar">AD</div><div className="profile-copy"><strong>Admin User</strong><small>System administrator</small></div><Activity size={16} className="profile-activity" />
      </div>
    </div>
  </aside>;
}