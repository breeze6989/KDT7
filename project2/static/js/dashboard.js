// static/js/dashboard.js
(() => {
  const grid = document.getElementById('gridContainer');
  const layoutBtn = document.getElementById('layoutBtn');
  const layoutMenu = document.getElementById('layoutMenu');

  // ===== 배열 드롭다운 =====
  if (layoutBtn && layoutMenu) {
    layoutBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      layoutMenu.classList.toggle('show');
    });
    window.addEventListener('click', () => layoutMenu.classList.remove('show'));
    layoutMenu.querySelectorAll('.dropdown-item').forEach(li => {
      li.addEventListener('click', () => {
        const layout = li.dataset.layout;
        grid.className = `grid-container grid-container-${layout}`;
        layoutBtn.textContent = `배열 ${li.textContent}`;
        layoutMenu.classList.remove('show');
      });
    });
  }

  // ===== 모달 =====
  let modal, modalVideo, modalCanvas, modalClose;
  function ensureModal() {
    if (modal) return;
    modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
      <div class="modal-content">
        <span class="close-button">×</span>
        <div class="video-box" style="width:100%;height:70vh">
          <video id="modalVideo" muted playsinline style="width:100%;height:100%;object-fit:contain;background:#000"></video>
          <canvas id="modalCanvas" class="box-layer"></canvas>
        </div>
      </div>`;
    document.body.appendChild(modal);
    modalVideo  = modal.querySelector('#modalVideo');
    modalCanvas = modal.querySelector('#modalCanvas');
    modalClose  = modal.querySelector('.close-button');
    modalClose.onclick = () => { modal.style.display='none'; modalVideo.pause(); modalVideo.src=''; };
    modal.addEventListener('click', (e)=>{ if(e.target===modal) modalClose.click(); });
  }
  function openModal(streamUrl, isHls, boxes) {
    ensureModal();

    // 모달 재생 (HLS/MP4)
    if (isHls) {
      if (window.Hls && Hls.isSupported()) {
        const hls = new Hls({ enableWorker:true });
        hls.loadSource(streamUrl);
        hls.attachMedia(modalVideo);
        hls.on(Hls.Events.MANIFEST_PARSED, ()=> modalVideo.play().catch(()=>{}));
      } else if (modalVideo.canPlayType('application/vnd.apple.mpegurl')) {
        modalVideo.src = streamUrl;
        modalVideo.addEventListener('loadedmetadata', ()=> modalVideo.play().catch(()=>{}), { once:true });
      } else {
        modalVideo.src = streamUrl;
        modalVideo.addEventListener('loadedmetadata', ()=> modalVideo.play().catch(()=>{}), { once:true });
      }
      // 안전장치: 끊김 시 재시도
      modalVideo.addEventListener('ended', () => modalVideo.play().catch(()=>{}));
    } else {
      // ★ 로컬 MP4는 반복 재생 보장
      modalVideo.loop = true;
      modalVideo.src = streamUrl;
      modalVideo.addEventListener('loadedmetadata', ()=> modalVideo.play().catch(()=>{}), { once:true });
      modalVideo.addEventListener('ended', ()=> { modalVideo.currentTime = 0; modalVideo.play().catch(()=>{}); });
    }

    modal.style.display = 'flex';
    const draw = () => {
      if (modal.style.display !== 'flex') return;
      const w = modalVideo.clientWidth, h = modalVideo.clientHeight;
      if (w && h) {
        if (modalCanvas.width !== w || modalCanvas.height !== h) {
          modalCanvas.width = w; modalCanvas.height = h;
        }
        const ctx = modalCanvas.getContext('2d');
        ctx.clearRect(0,0,w,h);
        (boxes || []).forEach(b=>{
          if (b.cls !== 0 && b.cls !== 2) return; // 불/연기만
          const x1=Math.round(b.x1*w), y1=Math.round(b.y1*h);
          const x2=Math.round(b.x2*w), y2=Math.round(b.y2*h);
          ctx.lineWidth=3; ctx.strokeStyle=(b.cls===0)?'#ff3b30':'#ffcc00';
          ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        });
      }
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  }

  // ===== 초기 데이터 로드 =====
  fetch('/api/dashboard_data').then(r => r.json()).then(data => {
    drawCards(data.cctv_list);
    fillSidebar(data.log_list);
  });

  function drawCards(cctvs){
    document.getElementById('totalCCTVs').textContent = cctvs.length;
    grid.innerHTML = '';
    cctvs.forEach((c,idx)=>{
      const card = document.createElement('div');
      card.className = 'grid-item';
      card.dataset.id = c.id;
      card.dataset.isHls = c.is_hls;

      card.innerHTML = `
        <span class="item-number">${idx+1}</span>
        <div class="video-box">
          <video muted playsinline></video>
          <canvas class="box-layer"></canvas>
          <div class="status-badge status-normal">정상</div>
        </div>
        <p class="cctv-name">${c.name}</p>
      `;

      const video  = card.querySelector('video');
      const canvas = card.querySelector('canvas');
      const isHls  = Number(c.is_hls) === 1;

      // 공통: 모바일/데스크톱 자동재생 보장
      video.muted = true;
      video.playsInline = true;

      // ===== 재생 설정 (핵심: 로컬 MP4는 loop 강제) =====
      if (isHls) {
        if (window.Hls && Hls.isSupported()) {
          const hls = new Hls({ enableWorker:true });
          hls.loadSource(c.stream_url);
          hls.attachMedia(video);
          hls.on(Hls.Events.MANIFEST_PARSED, ()=> video.play().catch(()=>{}));
        } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
          video.src = c.stream_url;
          video.addEventListener('loadedmetadata', ()=> video.play().catch(()=>{}), { once:true });
        } else {
          video.src = c.stream_url;
          video.addEventListener('loadedmetadata', ()=> video.play().catch(()=>{}), { once:true });
        }
        // HLS가 드물게 ended 나는 경우 재시작 시도
        video.addEventListener('ended', ()=> video.play().catch(()=>{}));
      } else {
        // ★ 로컬 MP4 반복 :: loop + ended 복구
        video.loop = true;
        video.src = c.stream_url;
        video.addEventListener('loadedmetadata', ()=> video.play().catch(()=>{}), { once:true });
        video.addEventListener('ended', ()=> { video.currentTime = 0; video.play().catch(()=>{}); });
      }

      // 카드 클릭 → 모달
      card.addEventListener('click', ()=>{
        openModal(c.stream_url, isHls, card.__lastBoxes || []);
      });

      grid.appendChild(card);
    });
    startRealtime();
  }

  function fillSidebar(logs){
    const ul = document.getElementById('sidebarLogUl');
    ul.innerHTML = '';
    logs.forEach(l=>{
      const li = document.createElement('li');
      li.textContent = `[${l.ts}] ${l.msg}`;
      ul.appendChild(li);
    });
  }

  // ===== 상태/박스 폴링 =====
  let pollTimer=null;
  function startRealtime(){
    if(pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async ()=>{
      try{
        const res = await fetch('/api/boxes_all',{ cache:'no-store' });
        if(!res.ok) return;
        const data = await res.json();
        const items = data.items || [];

        let fire=0, warn=0;
        items.forEach(it=>{
          const card = grid.querySelector(`.grid-item[data-id="${it.id}"]`);
          if(!card) return;
          const video = card.querySelector('video');
          const canvas= card.querySelector('canvas');
          const badge = card.querySelector('.status-badge');

          // 상태 배지
          badge.classList.remove('status-fire','status-warn','status-normal');
          if(it.status==='화재감지'){ badge.textContent='화재감지'; badge.classList.add('status-fire'); fire++; }
          else if(it.status==='주의'){ badge.textContent='주의'; badge.classList.add('status-warn'); warn++; }
          else { badge.textContent='정상'; badge.classList.add('status-normal'); }

          // 박스 그리기 + 모달 캐시
          drawBoxes(canvas, video, it.boxes);
          card.__lastBoxes = it.boxes || [];
        });

        document.getElementById('fireCnt').textContent = fire;
        document.getElementById('warnCnt').textContent = warn;
      }catch(_){}
    }, 250); // 필요 시 더 줄여도 됨
  }

  function drawBoxes(canvas, video, boxes){
    if(!video.videoWidth || !video.videoHeight) return;
    const w = video.clientWidth || video.videoWidth;
    const h = video.clientHeight || video.videoHeight;
    if(canvas.width!==w || canvas.height!==h){ canvas.width=w; canvas.height=h; }
    const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,w,h);
    (boxes || []).forEach(b=>{
      if(b.cls!==0 && b.cls!==2) return;  // object 등은 숨김
      const x1=Math.round(b.x1*w), y1=Math.round(b.y1*h);
      const x2=Math.round(b.x2*w), y2=Math.round(b.y2*h);
      ctx.lineWidth=3; ctx.strokeStyle=(b.cls===0)?'#ff3b30':'#ffcc00';
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    });
  }
})();



