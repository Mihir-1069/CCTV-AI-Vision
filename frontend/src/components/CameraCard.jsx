import { Camera, MoreHorizontal, Users, Video } from "lucide-react";

export default function CameraCard({ camera, onOpen }) {
  const online = camera.status === "online";
  return <button className="camera-card" onClick={() => onOpen?.(camera)}>
    <div className="camera-preview">
      <div className="preview-grid" />
      <div className="preview-label"><span className={online ? "live-dot" : "offline-dot"} />{online ? "LIVE" : "OFFLINE"}</div>
      <div className="preview-center"><Video size={30} /><span>{online ? "Video feed connected" : "No signal detected"}</span></div>
      <div className="preview-cam-id">{camera.id}</div>
    </div>
    <div className="camera-card-info">
      <div><strong>{camera.name}</strong><small>{camera.location}</small></div>
      <MoreHorizontal size={18} className="muted-icon" />
    </div>
    <div className="camera-card-footer"><span className="people-count"><Users size={14} />{camera.people} people</span><span className={online ? "status-online" : "status-offline"}>{online ? "Connected" : "Disconnected"}</span></div>
  </button>;
}