/* dashboard.js */
document.addEventListener('DOMContentLoaded', () => {
  const grid = document.getElementById('gridContainer');
  const btn  = document.getElementById('layoutToggleButton');
  const menu = document.getElementById('layoutDropdownMenu');

  btn.onclick = e => { e.stopPropagation(); menu.classList.toggle('show'); };
  window.addEventListener('click',()=> menu.classList.remove('show'));
  menu.querySelectorAll('.dropdown-item').forEach(it=>{
    it.onclick=()=>{
      grid.className='grid-container grid-container-'+it.dataset.layout;
      btn.textContent='배열 '+it.textContent;
      menu.classList.remove('show');
    };
  });

  fetch('/api/dashboard_data').then(r=>r.json()).then(render);

  function render(d){
    document.getElementById('totalCCTVs').textContent = d.cctv_list.length;
    document.getElementById('fireDetections').textContent =
       d.cctv_list.filter(c=>c.status==='화재감지').length;

    grid.innerHTML='';
    d.cctv_list.forEach(c=>{
      const card=document.createElement('div'); card.className='grid-item';
      Object.assign(card.dataset,{
        stream:c.stream_url, lat:c.lat, lng:c.lng,
        status:c.status , cctvid:c.id , msg:`CCTV #${c.id} (${c.status})`
      });
      card.innerHTML=`<span class="item-number">${c.id}</span>
        <span class="status ${c.status}">${c.status}</span>
        <video muted src="${c.stream_url}"></video>
        <p class="cctv-name">${c.name}</p>`;
      card.querySelector('video').onloadedmetadata=e=>e.target.play();
      grid.appendChild(card);
    });

    const ul=document.getElementById('sidebarLogUl'); ul.innerHTML='';
    d.log_list.forEach(l=>{
      const li=document.createElement('li'); li.className='sidebar-log';
      Object.assign(li.dataset,{
        stream:l.stream_url,lat:l.lat,lng:l.lng,
        status:'화재감지',cctvid:l.cctv_id,msg:l.msg
      });
      li.textContent=`[${l.ts}] ${l.msg}`; ul.appendChild(li);
    });
  }
});
