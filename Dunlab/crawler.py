"""Crawl dundam damage_ranking pages and upsert characters."""
import time, requests
from bs4 import BeautifulSoup
from database import SessionLocal
from models import Character
from neople_api import find_character_id, fetch_character_detail, extract_stats

SERVER_MAP={"카인":"cain","프레이":"prey","바칼":"bakal","디레지에":"diregie","시로코":"siroco","카시야스":"casillas"}
HEADERS={"User-Agent":"Mozilla/5.0"}

def _page(p:int):
    url=f"https://dundam.xyz/damage_ranking?page={p}&type=7"
    soup=BeautifulSoup(requests.get(url,headers=HEADERS,timeout=10).text,"html.parser")
    names=[t.text.strip() for t in soup.select(".nik")]
    servers=[t.text.strip() for t in soup.select(".svname")]
    return list(zip(names,servers))

def populate_from_dundam(max_page:int=3):
    db=SessionLocal()
    try:
        for p in range(1,max_page+1):
            for n,sv in _page(p):
                sv_eng=SERVER_MAP.get(sv); time.sleep(0.2)
                if not sv_eng: continue
                cid=find_character_id(sv_eng,n)
                if not cid: continue
                char=db.query(Character).filter_by(server=sv_eng,character_id=cid).first()
                stats=extract_stats(fetch_character_detail(sv_eng,cid))
                if char:
                    for k,v in stats.items(): setattr(char,k,v)
                else:
                    db.add(Character(server=sv_eng,character_id=cid,name=n,**stats))
            db.commit()
    finally:
        db.close()
populate_from_dundam(10)