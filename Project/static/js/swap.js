/* swap.js ── 영상 교체 모드 (1-선택 → 다중 대상 → Enter 교체 / Esc 복구) */
document.addEventListener('DOMContentLoaded', () => {

  let swapMode   = false;    // 교체모드 on/off
  let srcCard    = null;     // 원본 카드
  const originals = new Map(); // 대상 카드 → 원본 src

  const btn  = document.getElementById('swapModeBtn');
  const grid = document.getElementById('gridContainer');

  /* ── 모드 토글 버튼 ─────────────────────────────────────── */
  function toggleMode(on){
    swapMode = on;
    window.swapMode = swapMode;          // modal.js 와 공유
    btn.textContent = on ? '교체 모드 종료(Esc)' : '영상 교체 모드';
    btn.classList.toggle('active', on);

    // 표시 초기화
    grid.querySelectorAll('.swap-src, .swap-tgt').forEach(c=>{
      c.classList.remove('swap-src','swap-tgt');
      if(originals.has(c)){
        c.querySelector('video').src = originals.get(c); // 복원
      }
    });
    srcCard  = null;
    originals.clear();
  }
  btn.onclick = () => toggleMode(!swapMode);

  /* ── 카드 클릭 흐름 ─────────────────────────────────────── */
  grid.addEventListener('click', e => {
    if(!swapMode) return;

    const card = e.target.closest('.grid-item');
    if(!card) return;

    // ① 원본 지정
    if(!srcCard){
      srcCard = card;
      card.classList.add('swap-src');
      return;
    }
    // ② 대상 토글
    if(card !== srcCard){
      card.classList.toggle('swap-tgt');
      if(card.classList.contains('swap-tgt')){
        originals.set(card, card.querySelector('video').src); // 백업
      }else{
        card.querySelector('video').src = originals.get(card); // 복원
        originals.delete(card);
      }
    }
  });

  /* ── 키보드 단축 ─────────────────────────────────────────── */
  window.addEventListener('keydown', e => {
    if(!swapMode) return;

    if(e.key === 'Enter' && srcCard && originals.size){
      const newSrc = srcCard.querySelector('video').src;
      originals.forEach((_, card) => {
        card.querySelector('video').src = newSrc;
        card.dataset.stream = newSrc;          // 모달에서도 교체
        card.classList.remove('swap-tgt');
      });
      originals.clear();
      alert('교체 완료');
      toggleMode(false);                       // 자동 종료
    }
    if(e.key === 'Escape'){                    // 취소
      toggleMode(false);
    }
  });
});
