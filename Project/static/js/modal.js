// modal.js – 대시보드 · 로그 공용 모달
let leafletMap = null;

/* ── 스트림 연결 ─────────────────────────────── */
function attachStream(el, url){
  if (Hls.isSupported() && url.endsWith('.m3u8')){
    const hls = new Hls();
    hls.loadSource(url); hls.attachMedia(el);
  } else el.src = url;
}

/* ── 모달 오픈 ──────────────────────────────── */
function openModal(el){
  /* 데이터 추출 */
  const { stream, lat, lng, cctvid, status,
          area, last_ts, today_cnt } = el.dataset;

  /* 비디오 */
  const modal = document.getElementById('cctvModal');
  const video = document.getElementById('modalVideo');
  attachStream(video, stream);

  /* 지도 */
  if (!leafletMap){
    leafletMap = L.map('modalMap').setView([lat, lng], 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution:'© OpenStreetMap' }).addTo(leafletMap);
  } else {
    leafletMap.setView([lat, lng], 15);
    leafletMap.eachLayer(l=>{ if(l instanceof L.Marker) leafletMap.removeLayer(l); });
  }
  L.marker([lat,lng]).addTo(leafletMap)
   .bindPopup(`CCTV #${cctvid}<br>${status}`).openPopup();
  requestAnimationFrame(()=>leafletMap.invalidateSize());

  /* ✅ 오른쪽 하단 정보 삽입 */
  document.getElementById('modalInfo').innerHTML = `
    CCTV ID: 4<br>
    상태: 화재 감지<br>
    설치 지역: 중구<br>
    최근 감지 시간: 2025-07-29 10:30<br>
    경보 횟수(오늘): 3회`;

  modal.style.display='flex';
}

/* ── 전역 델리게이트 ────────────────────────── */
document.addEventListener('DOMContentLoaded', ()=>{
  /* (1) 카드·로그 클릭 */
  document.body.addEventListener('click', e=>{
    const t=e.target.closest('.grid-item,.sidebar-log li,[data-stream]');
    if(!t || !t.dataset.stream || window.swapMode) return;
    openModal(t);
  });

  /* (2) 닫기 */
  const modal = document.getElementById('cctvModal');
  document.querySelector('.close-button').onclick = () => {
    modal.style.display='none'; document.getElementById('modalVideo').pause();
  };
  modal.onclick = e=>{ if(e.target===modal){ modal.style.display='none';
                          document.getElementById('modalVideo').pause(); } };
});
