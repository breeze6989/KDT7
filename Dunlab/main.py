from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import or_

from database import Base, engine, get_db
from models import Character
from neople_api import find_character_id, fetch_character_detail, extract_stats, search_characters
from scheduler import start_scheduler

PAGE_SIZE = 10
BUFFER_JOBS = ["크루세이더","인챈트리스","뮤즈"]
is_buffer = lambda j: any(k in j for k in BUFFER_JOBS)

@asynccontextmanager
async def lifespan(app:FastAPI):
    start_scheduler(); yield

app = FastAPI(title="Dunlab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
Base.metadata.create_all(bind=engine)

# -------------------------------------------------- 홈
@app.get("/", response_class=HTMLResponse)
async def home(r:Request, db:Session=Depends(get_db)):
    recent = (db.query(Character)
                .order_by(Character.updated_at.desc())
                .limit(10).all())
    return templates.TemplateResponse("search.html",
                                      {"request": r, "recent": recent})

# ---------- 검색 ----------
@app.post("/search", response_class=HTMLResponse)
async def search_character(
    r:Request, server:str=Form(...), name:str=Form(...), db:Session=Depends(get_db)
):
    # 1) 모험단 검색 -----------------
    if server == "모험단":
        rows = (db.query(Character)
                  .filter(Character.adventure_name == name)
                  .order_by(Character.fame.desc()).all())
        return templates.TemplateResponse("search.html",
                                          {"request": r, "chars": rows})

    # 2) 통합 검색 ------------------
    if server in ("all", "", "전체"):
        rows = search_characters("all", name, wordType="full", limit=20)
        if not rows:
            raise HTTPException(404, "검색 결과 없음")

        found_chars = []
        for row in rows:
            sv = row["serverId"]
            cid = row["characterId"]
            detail = fetch_character_detail(sv, cid)
            stats  = extract_stats(detail)
            adv    = detail["basic"].get("adventureName")
            ch = (db.query(Character)
                    .filter_by(server=sv, character_id=cid).first())
            if not ch:
                ch = Character(server=sv, character_id=cid,
                               name=row["characterName"],
                               adventure_name=adv, **stats)
                db.add(ch)
            else:
                ch.adventure_name = adv
                for k,v in stats.items(): setattr(ch,k,v)
            db.flush()
            found_chars.append(ch)
        db.commit()
        return templates.TemplateResponse("search.html",
                                          {"request": r, "chars": found_chars})

    # 3) 단일 서버 검색 (기존 로직) -----
    cid = find_character_id(server, name, wordType="full")
    if not cid:
        raise HTTPException(404, "캐릭터를 찾지 못했습니다.")

    detail = fetch_character_detail(server, cid)
    stats  = extract_stats(detail)
    adv    = detail["basic"].get("adventureName")

    ch = (db.query(Character)
            .filter_by(server=server, character_id=cid).first())
    if not ch:
        ch = Character(server=server, character_id=cid,
                    name=name, adventure_name=adv, **stats)
        db.add(ch)
    else:
        ch.adventure_name = adv
        for k,v in stats.items(): setattr(ch, k, v)
    db.commit(); db.refresh(ch)
    
    return templates.TemplateResponse(
    "search.html",
    {"request": r, "chars": [ch]}   # 리스트에 단일 객체
)
# -------------------------------------------------- 상세
@app.get("/character/{cid}",response_class=HTMLResponse)
async def detail(cid:int,r:Request,db:Session=Depends(get_db)):
    ch=db.get(Character,cid) or HTTPException(404,"없음")
    power = ch.buff_power*1.3 if is_buffer(ch.job_name) else ch.final_damage*1.5
    return templates.TemplateResponse("character.html",
                                      {"request":r,"char":ch,"power":power,
                                       "is_buffer":is_buffer(ch.job_name)})

# -------------------------------------------------- 랭킹 공통
def _base(db,buf):                                  # buf=True→버퍼
    filt=or_(*[Character.job_name.contains(j) for j in BUFFER_JOBS])
    q=db.query(Character).filter(filt) if buf else db.query(Character).filter(~filt)
    order = Character.buff_power.desc() if buf else Character.final_damage.desc()
    return q.order_by(order)

# -------------------------------------------------- 딜러 랭킹
@app.get("/damage_ranking",response_class=HTMLResponse)
async def dmg_rank(r:Request,page:int=1,db:Session=Depends(get_db)):
    base=_base(db,False)
    rows=base.offset((page-1)*PAGE_SIZE).limit(PAGE_SIZE).all()
    total=base.count()
    return templates.TemplateResponse("ranking.html",
        {"request":r,"chars":rows,"mode":"damage","page":page,"total":total,"size":PAGE_SIZE})

# -------------------------------------------------- 버퍼 랭킹
@app.get("/buff_ranking",response_class=HTMLResponse)
async def buf_rank(r:Request,page:int=1,db:Session=Depends(get_db)):
    base=_base(db,True)
    rows=base.offset((page-1)*PAGE_SIZE).limit(PAGE_SIZE).all()
    total=base.count()
    return templates.TemplateResponse("ranking.html",
        {"request":r,"chars":rows,"mode":"buff","page":page,"total":total,"size":PAGE_SIZE})

