// static/js/modal.js
(() => {
  const modal = document.getElementById('cctvModal');
  const video = document.getElementById('modalVideo');
  const canvas = document.getElementById('modalCanvas'); // 모달 내부 캔버스가 있어야 함
  const closeBtn = document.querySelector('#cctvModal .close-button');

  let poll = null;
  let currentCid = null;

  function attachHls(el, src){
    try{
      if (Hls.isSupported() && src.includes('.m3u8')) {
        const hls = new Hls({
          lowLatencyMode: true, liveSyncDurationCount: 1, maxBufferLength: 3
        });
        hls.loadSource(src);
        hls.attachMedia(el);
        hls.on(Hls.Events.MANIFEST_PARSED, ()=> el.play().catch(()=>{}));
      } else {
        el.src = src; el.play().catch(()=>{});
      }
    } catch(_){}
  }

  function drawBoxes(boxes){
    if (!video.clientWidth || !video.clientHeight) return;
    const W = video.clientWidth, H = video.clientHeight;
    if (canvas.width !== W || canvas.height !== H){
      canvas.width = W; canvas.height = H;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);

    (boxes||[]).forEach(b=>{
      if (b.cls !== 0 && b.cls !== 2) return;
      const x1 = Math.round(b.x1 * W), y1 = Math.round(b.y1 * H);
      const x2 = Math.round(b.x2 * W), y2 = Math.round(b.y2 * H);
      ctx.lineWidth = 3;
      ctx.strokeStyle = (b.cls === 0) ? '#ff3b30' : '#ffcc00';
      ctx.strokeRect(x1, y1, Math.max(1, x2-x1), Math.max(1, y2-y1));
    });
  }

  async function startModalWatch(cid){
    await fetch('/api/modal_start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ cid })
    });

    if (poll) clearInterval(poll);
    poll = setInterval(async ()=>{
      const r = await fetch(`/api/modal_boxes?cid=${cid}`, { cache: 'no-store' });
      if (!r.ok) return;
      const data = await r.json();
      drawBoxes(data.boxes);
    }, 200);
  }

  async function stopModalWatch(){
    if (poll) { clearInterval(poll); poll = null; }
    if (currentCid != null){
      await fetch('/api/modal_stop', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ cid: currentCid })
      });
      currentCid = null;
    }
  }

  // 전역에서 호출
  window.openModal = ({ id, name, stream })=>{
    currentCid = id;
    attachHls(video, stream);
    startModalWatch(id);
    modal.style.display = 'flex';
  };

  function closeModal(){
    stopModalWatch();
    video.pause(); video.removeAttribute('src'); video.load();
    modal.style.display = 'none';
  }

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', (e)=>{
    if (e.target === modal) closeModal();
  });
})();
