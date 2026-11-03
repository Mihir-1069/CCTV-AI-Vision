import { useState } from "react";
import { Bell, Database, LockKeyhole, Save, SlidersHorizontal } from "lucide-react";

export default function Settings() {
  const [threshold, setThreshold] = useState(20);
  const [retention, setRetention] = useState("7");
  const [saved, setSaved] = useState(false);
  return <div className="page-content settings-page">
    <div className="page-toolbar"><div><h2>Workspace settings</h2><p>Configure demo preferences and review safeguards.</p></div></div>
    <section className="panel settings-panel"><div className="settings-section-title"><div className="settings-icon"><SlidersHorizontal size={17}/></div><div><h3>Monitoring preferences</h3><p>Basic configuration for the prototype.</p></div></div>
      <div className="setting-row"><div><strong>Occupancy threshold</strong><p>Flag a zone when its estimated count exceeds this value.</p></div><div className="setting-control"><input type="number" min="1" max="200" value={threshold} onChange={e=>setThreshold(e.target.value)}/><span>people</span></div></div>
      <div className="setting-row"><div><strong>Evidence retention</strong><p>Demo setting only; production retention should be policy-defined.</p></div><select className="select-control" value={retention} onChange={e=>setRetention(e.target.value)}><option value="1">1 day</option><option value="7">7 days</option><option value="30">30 days</option></select></div>
    </section>
    <section className="panel settings-panel"><div className="settings-section-title"><div className="settings-icon"><Bell size={17}/></div><div><h3>Alert review</h3><p>Human review is required before treating an AI event as confirmed.</p></div></div>
      <div className="setting-row"><div><strong>Human review required</strong><p>AI-generated event candidates remain unconfirmed until reviewed.</p></div><span className="enabled-pill">Enabled</span></div>
    </section>
    <section className="panel settings-panel"><div className="settings-section-title"><div className="settings-icon"><Database size={17}/></div><div><h3>Data and privacy</h3><p>Use least-privilege access and avoid storing unnecessary raw footage.</p></div></div>
      <div className="setting-row"><div><strong>Anonymous tracking</strong><p>Use temporary track IDs rather than facial identification.</p></div><span className="enabled-pill">Default</span></div>
      <div className="setting-row"><div><strong>Access controls</strong><p>Connect authentication and role-based permissions in the backend.</p></div><span className="enabled-pill">Required</span></div>
    </section>
    <button className="primary-button save-button" onClick={()=>{setSaved(true);setTimeout(()=>setSaved(false),2500)}}><Save size={15}/>{saved ? "Saved demo preferences" : "Save preferences"}</button>
  </div>;
}