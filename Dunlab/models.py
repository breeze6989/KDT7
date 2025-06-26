from sqlalchemy import Column, Integer, String, Float, DateTime, func, UniqueConstraint
from database import Base

class Character(Base):
    __tablename__ = "characters"
    __table_args__ = (UniqueConstraint("server","character_id", name="uix_server_character_id"),)

    id                 = Column(Integer, primary_key=True, index=True)
    server             = Column(String, index=True)
    character_id       = Column(String, index=True)
    name               = Column(String, index=True)
    adventure_name     = Column(String, index=True)   # 모험단
    job_name           = Column(String, index=True)
    job_id             = Column(String)
    level              = Column(Integer)
    fame               = Column(Integer)

    # 핵심 지표
    final_damage       = Column(Float)
    buff_power         = Column(Float)
    cooldown_reduction = Column(Float)

    # 추가 스탯
    strength           = Column(Integer)
    intelligence       = Column(Integer)
    vitality           = Column(Integer)
    physical_attack    = Column(Integer)
    magic_attack       = Column(Integer)
    independent_attack = Column(Integer)

    physical_def       = Column(Float)
    magical_def        = Column(Float)
    crit_phy           = Column(Float)
    crit_mag           = Column(Float)
    attack_speed       = Column(Float)
    cast_speed         = Column(Float)

    fire_ele           = Column(Integer)
    water_ele          = Column(Integer)
    light_ele          = Column(Integer)
    dark_ele           = Column(Integer)

    damage_inc         = Column(Float)
    cool_reduce        = Column(Float)

    updated_at         = Column(DateTime(timezone=True),
                                 server_default=func.now(),
                                 onupdate=func.now())
