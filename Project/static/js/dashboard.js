/* dashboard.js – 레이아웃 · 통계 · 로그 · 상태 폴링 */
document.addEventListener('DOMContentLoaded', () => {

  const grid = document.getElementById('gridContainer');
  const btn  = document.getElementById('layoutBtn');
  const menu = document.getElementById('layoutMenu');

  /* ── 배열 드롭다운 ─────────────────────────── */
  btn.onclick = e => { e.stopPropagation(); menu.classList.toggle('show'); };
  window.addEventListener('click', () => menu.classList.remove('show'));
  menu.querySelectorAll('li').forEach(li=>{
    li.onclick = () => {
      grid.className = 'grid-container grid-container-'+li.dataset.layout;
      btn.textContent = '배열 ' + li.textContent.trim();
      menu.classList.remove('show');
    };
  });

  /* ── 최초 렌더 ─────────────────────────────── */
  function render(d){
    /* 상단 통계 */
    document.getElementById('totalCCTVs').textContent = d.cctv_list.length;
    document.getElementById('fireCnt').textContent    =
        d.cctv_list.filter(c=>c.status==='화재감지').length;
    document.getElementById('warnCnt').textContent    =
        d.cctv_list.filter(c=>c.status==='주의').length;

    /* 카드 그리드 */
    grid.innerHTML='';
    d.cctv_list.forEach(c=>{
      const card=document.createElement('div');
      card.className='grid-item';
      Object.entries(c).forEach(([k,v])=>card.dataset[k]=v);   // 모든 필드 data-속성
      card.dataset.stream = c.stream_url;                      // 모달용

      card.innerHTML=`
        <span class="item-number">${c.id}</span>
        <span class="status ${c.status}">${c.status}</span>
        <video muted loop src="${c.stream_url}"></video>
        <p class="cctv-name">${c.name}</p>`;

      const v = card.querySelector('video');
      v.addEventListener('loadedmetadata', ()=>v.play());

      grid.appendChild(card);
    });

    /* 사이드바 최신 5 로그 */
    const ul=document.getElementById('sidebarLogUl'); ul.innerHTML='';
    d.log_list.slice(0,5).forEach(l=>{
      const li=document.createElement('li');
      li.className='sidebar-log li';
      Object.entries(l).forEach(([k,v])=>li.dataset[k]=v);
      li.dataset.stream = l.stream_url;
      li.textContent=`[${l.ts}] ${l.msg}`;
      ul.appendChild(li);
    });
  }
  fetch('/api/dashboard_data').then(r=>r.json()).then(render);

  /* ── 5초 주기 상태-라벨 폴링 ───────────────── */
  function diffUpdate(d){
    d.cctv_list.forEach(c=>{
      const card = document.querySelector(`.grid-item[data-id='${c.id}']`);
      if(!card) return;
      if(card.dataset.status !== c.status){
        card.dataset.status = c.status;
        const s = card.querySelector('.status');
        s.textContent = c.status;
        s.className   = 'status '+c.status;
      }
    });
    document.getElementById('fireCnt').textContent =
        d.cctv_list.filter(c=>c.status==='화재감지').length;
    document.getElementById('warnCnt').textContent =
        d.cctv_list.filter(c=>c.status==='주의').length;
  }
  setInterval(()=>fetch('/api/dashboard_data')
               .then(r=>r.json()).then(diffUpdate), 500);
});
