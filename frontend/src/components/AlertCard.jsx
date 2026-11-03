import { ArrowUpRight, Clock3 } from "lucide-react";

export default function AlertCard({ alert, compact = false, onReview }) {
  return <div className={`alert-row ${compact ? "compact" : ""}`}>
    <div className={`alert-symbol ${alert.severity}`}><ArrowUpRight size={17} /></div>
    <div className="alert-content"><div className="alert-title-line"><strong>{alert.type}</strong><span className={`severity ${alert.severity}`}>{alert.severity}</span></div><p>{alert.camera} <span>·</span> {alert.time}</p>{!compact && <small>{alert.description}</small>}</div>
    {onReview && <button className="text-action" onClick={() => onReview(alert)}>Review</button>}
  </div>;
}