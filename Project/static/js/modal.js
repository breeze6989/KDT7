/* modal.js */
let leafletMap=null;
function attachStream(el,u){ if(Hls.isSupported()&&u.endsWith('.m3u8')){
  const h=new Hls();h.loadSource(u);h.attachMedia(el);}else{el.src=u;} }

function openModal(el){
  const {stream,lat,lng,msg,cctvid,status}=el.dataset;
  attachStream(document.getElementById('modalVideo'),stream);

  if(!leafletMap){
    leafletMap=L.map('modalMap').setView([lat,lng],15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      {attribution:'© OpenStreetMap'}).addTo(leafletMap);
  }else{
    leafletMap.setView([lat,lng],15);
    leafletMap.eachLayer(l=>{if(l instanceof L.Marker)leafletMap.removeLayer(l);});
  }
  L.marker([lat,lng]).addTo(leafletMap).bindPopup(msg).openPopup();
  requestAnimationFrame(()=>leafletMap.invalidateSize());

  document.getElementById('modalInfo').innerHTML=
    `<strong>${msg}</strong><br>좌표: ${(+lat).toFixed(5)}, ${(+lng).toFixed(5)}`;

  const btn=document.getElementById('ackButton');
  btn.style.display=status==='화재감지'?'block':'none';
  btn.onclick=()=>fetch('/api/ack_alert',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cctv_id:cctvid})}).then(()=>{btn.style.display='none';alert('경보 해제 완료');});
  document.getElementById('cctvModal').style.display='flex';
}

document.addEventListener('DOMContentLoaded',()=>{
  document.body.addEventListener('click',e=>{
    const el=e.target.closest('[data-stream]'); if(el) openModal(el);
  });
  const modal=document.getElementById('cctvModal');
  document.querySelector('.close-button').onclick=()=>modal.style.display='none';
  modal.onclick=e=>{if(e.target===modal)modal.style.display='none';};
});
