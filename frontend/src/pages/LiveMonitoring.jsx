import { useEffect, useState } from "react";
import { Maximize2, SlidersHorizontal, Users, Video } from "lucide-react";
import { getCameras } from "../services/api";

export default function LiveMonitoring() {
  const [cameras, setCameras] = useState([]);
  const [selected, setSelected] = useState("all");
  useEffect(() => { getCameras().then(setCameras).catch(console.error); }, []);
  const filtered = cameras.filter(c => selected === "all" || c.status === selected);
  return <div className="page-content">
    <div className="page-toolbar"><div><h2>Camera feeds</h2><p>Streams are represented with preview placeholders in demo mode.</p></div><div className="toolbar-actions"><select value={selected} onChange={e => setSelected(e.target.value)} className="select-control"><option value="all">All cameras</option><option value="online">Online</option><option value="offline">Offline</option></select><button className="secondary-button"><SlidersHorizontal size={15}/> Filter</button></div></div>
    <div className="live-grid">{filtered.map(c => <div className="live-tile" key={c.id}>
      <div className="live-video"><div className="preview-grid"/><span className={`feed-badge ${c.status}`}>{c.status === "online" ? "● LIVE" : "● OFFLINE"}</span><div className="feed-placeholder"><Video size={34}/><span>{c.status === "online" ? "Camera feed preview" : "Stream unavailable"}</span><small>Connect your camera stream in the backend to display video.</small></div><button className="expand-button"><Maximize2 size={15}/></button><span className="feed-people"><Users size={13}/>{c.people}</span></div>
      <div className="live-tile-info"><div><strong>{c.name}</strong><small>{c.id} · {c.location}</small></div><span className={c.status === "online" ? "status-online" : "status-offline"}>{c.status}</span></div>
    </div>)}</div>
  </div>;
}