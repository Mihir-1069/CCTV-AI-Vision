import { useEffect, useState } from "react";
import { Camera, Plus, Search, Wifi, WifiOff } from "lucide-react";
import { getCameras } from "../services/api";

export default function CameraManagement() {
  const [cameras, setCameras] = useState([]);
  const [search, setSearch] = useState("");
  useEffect(() => { getCameras().then(setCameras).catch(console.error); }, []);
  const filtered = cameras.filter(c => `${c.name} ${c.location} ${c.id}`.toLowerCase().includes(search.toLowerCase()));
  return <div className="page-content">
    <div className="page-toolbar"><div><h2>Registered cameras <span className="count-badge">{cameras.length}</span></h2><p>Manage camera sources and connection status.</p></div><button className="primary-button"><Plus size={16}/> Add camera</button></div>
    <div className="panel table-panel"><div className="table-toolbar"><div className="table-search"><Search size={15}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search cameras..." /></div><span className="table-caption">Showing {filtered.length} cameras</span></div>
      <div className="table-scroll"><table><thead><tr><th>Camera</th><th>Camera ID</th><th>Location</th><th>Status</th><th>People</th><th>FPS</th><th>Actions</th></tr></thead><tbody>{filtered.map(c => <tr key={c.id}><td><div className="table-camera"><div className="table-camera-icon"><Camera size={16}/></div><strong>{c.name}</strong></div></td><td className="mono">{c.id}</td><td>{c.location}</td><td><span className={`table-status ${c.status}`}>{c.status === "online" ? <Wifi size={13}/> : <WifiOff size={13}/>} {c.status}</span></td><td>{c.people}</td><td>{c.fps || "—"}</td><td><button className="table-action">Configure</button></td></tr>)}</tbody></table></div>
    </div>
    <div className="info-banner"><div className="info-dot">i</div><div><strong>Camera configuration is in demo mode</strong><p>Camera registration and RTSP credentials will be managed by the FastAPI backend. Avoid placing stream passwords in frontend code.</p></div></div>
  </div>;
}