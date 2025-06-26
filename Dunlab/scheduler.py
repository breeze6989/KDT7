from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Character
from neople_api import fetch_character_detail, extract_stats
from crawler import populate_from_dundam

scheduler=BackgroundScheduler()

def refresh_all():
    db=SessionLocal()
    for c in db.query(Character).all():
        for k,v in extract_stats(fetch_character_detail(c.server,c.character_id)).items(): setattr(c,k,v)
    db.commit(); db.close()

def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(refresh_all,"cron",hour=4)
        scheduler.add_job(populate_from_dundam,"cron",hour=3)
        scheduler.start()