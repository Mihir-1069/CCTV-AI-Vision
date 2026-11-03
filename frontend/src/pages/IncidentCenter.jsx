import { useEffect, useState } from "react";
import { Check, Search, ShieldAlert } from "lucide-react";
import { getAlerts } from "../services/api";
import AlertCard from "../components/AlertCard";

export default function IncidentCenter() {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  useEffect(() => { getAlerts().then(setAlerts).catch(console.error); }, []);
  const review = (item) => setAlerts(prev => prev.map(a => a.id === item.id ? { ...a, status: "Reviewed" } : a));
  const visible = alerts.filter(a => (filter === "all" || a.status.toLowerCase().includes(filter)) && `${a.type} ${a.camera} ${a.id}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="page-content">
    <div className="page-toolbar"><div><h2>Event review queue</h2><p>AI outputs are candidate events. Review before taking action.</p></div><div className="incident-summary"><ShieldAlert size={17}/><span>{alerts.filter(a => a.status !== "Reviewed").length} awaiting review</span></div></div>
    <div className="panel incident-panel"><div className="incident-filters"><div className="table-search"><Search size={15}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search incidents..." /></div><div className="filter-tabs">{["all","pending","reviewed"].map(f => <button key={f} className={filter === f ? "selected" : ""} onClick={() => setFilter(f)}>{f === "all" ? "All events" : f === "pending" ? "Pending review" : "Reviewed"}</button>)}</div></div>
      <div className="incident-list">{visible.map(a => <div className="incident-item" key={a.id}><div className="incident-id">{a.id}</div><AlertCard alert={a}/><div className="incident-review"><span className={`review-status ${a.status === "Reviewed" ? "done" : ""}`}>{a.status}</span><button disabled={a.status === "Reviewed"} onClick={() => review(a)} className="review-button">{a.status === "Reviewed" ? <><Check size={14}/> Reviewed</> : "Mark reviewed"}</button></div></div>)}</div>
    </div>
  </div>;
}