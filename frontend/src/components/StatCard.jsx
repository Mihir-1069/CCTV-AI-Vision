import { ArrowUpRight } from "lucide-react";

export default function StatCard({ label, value, change, icon: Icon, tone = "blue", note }) {
  return <div className="stat-card">
    <div className="stat-top"><span>{label}</span><div className={`stat-icon ${tone}`}><Icon size={18} /></div></div>
    <div className="stat-value">{value}</div>
    <div className="stat-bottom">{change && <span className="trend"><ArrowUpRight size={13} />{change}</span>}<span>{note}</span></div>
  </div>;
}