import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getOccupancyAnalytics } from "../services/api";

export default function Analytics() {
  const [data, setData] = useState([]);
  useEffect(() => { getOccupancyAnalytics().then(setData).catch(console.error); }, []);
  const zones = [{ name: "Block A", people: 42 }, { name: "Block B", people: 18 }, { name: "Block C", people: 26 }, { name: "Sports", people: 21 }];
  return <div className="page-content">
    <div className="page-toolbar"><div><h2>Campus analytics</h2><p>Aggregated occupancy trends from monitored zones.</p></div><select className="select-control"><option>Today</option><option>This week</option><option>This month</option></select></div>
    <div className="analytics-grid">
      <section className="panel analytics-panel"><div className="panel-heading"><div><h3>Occupancy by hour</h3><p>Estimated people detected over time</p></div></div><div className="analytics-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={data}><CartesianGrid stroke="#202a3c" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="time" tick={{fill:"#7f8ba3",fontSize:11}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#7f8ba3",fontSize:11}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#111a2b",border:"1px solid #2a3850",borderRadius:10}}/><Bar dataKey="count" fill="#6b96ff" radius={[5,5,0,0]}/></BarChart></ResponsiveContainer></div></section>
      <section className="panel analytics-panel"><div className="panel-heading"><div><h3>Occupancy by zone</h3><p>Current estimated distribution</p></div></div><div className="zone-list">{zones.map((z,i)=><div className="zone-row" key={z.name}><div className="zone-label"><span className={`zone-index z${i}`}>{String(i+1).padStart(2,"0")}</span><strong>{z.name}</strong><b>{z.people}</b></div><div className="zone-bar"><i style={{width:`${z.people/45*100}%`}}/></div></div>)}</div></section>
    </div>
    <div className="info-banner"><div className="info-dot">i</div><div><strong>Analytics are estimates</strong><p>Counts depend on camera placement, detection quality, occlusion and configured zones. Use these metrics for operational review, not individual student assessment.</p></div></div>
  </div>;
}