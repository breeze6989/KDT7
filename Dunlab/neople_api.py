from typing import Dict, Any, Optional
import requests

API_KEY = "TXmmtt1yVmrB02RHlHs7W7T1ZQAOtuQx"
BASE    = "https://api.neople.co.kr/df"
HEADERS = {"User-Agent":"Dunlab-App"}

def _get(path:str, **params):
    params["apikey"] = API_KEY
    r = requests.get(f"{BASE}{path}", params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()

def find_character_id(server:str, name:str, wordType:str="full") -> Optional[str]:
    rows = _get(f"/servers/{server}/characters",
                characterName=name, wordType=wordType, limit=1).get("rows", [])
    return rows[0]["characterId"] if rows else None

def fetch_character_detail(server:str, cid:str)->Dict[str,Any]:
    return {
        "basic" : _get(f"/servers/{server}/characters/{cid}"),
        "status": _get(f"/servers/{server}/characters/{cid}/status"),
    }

def _lst2dict(lst): return {i["name"]:i.get("value") for i in lst}

def extract_stats(payload:Dict[str,Any]) -> Dict[str,Any]:
    basic = payload["basic"]
    stat_list = payload["status"].get("status") if isinstance(payload["status"],dict) else payload["status"]
    s = _lst2dict(stat_list)

    str_, int_, vit = s.get("힘",0), s.get("지능",0), s.get("체력",0)
    patk, matk, iatk = s.get("물리 공격",0), s.get("마법 공격",0), s.get("독립 공격",0)

    return {
        "level"       : basic.get("level"),
        "job_name"    : basic.get("jobGrowName", basic.get("jobName")),
        "job_id"      : basic.get("jobGrowId", basic.get("jobId")),
        "fame"        : basic.get("fame",0),
        "strength"    : str_,
        "intelligence": int_,
        "vitality"    : vit,
        "physical_attack" : patk,
        "magic_attack"    : matk,
        "independent_attack": iatk,

        "physical_def": s.get("물리 방어율"),
        "magical_def" : s.get("마법 방어율"),
        "crit_phy"    : s.get("물리 크리티컬"),
        "crit_mag"    : s.get("마법 크리티컬"),
        "attack_speed": s.get("공격 속도"),
        "cast_speed"  : s.get("캐스팅 속도"),

        "fire_ele"  : s.get("화속성 강화"),
        "water_ele" : s.get("수속성 강화"),
        "light_ele" : s.get("명속성 강화"),
        "dark_ele"  : s.get("암속성 강화"),

        "damage_inc" : s.get("공격력 증가"),
        "cool_reduce": s.get("쿨타임 감소"),

        "final_damage": s.get("최종 데미지 증가",0),
        "buff_power"  : s.get("버프력",0),
        "cooldown_reduction": s.get("쿨타임 감소",0),
    }
    
def search_characters(server: str, name: str,
                      wordType: str = "full", limit: int = 20):
    """
    server='all' 이면 전 서버 통합 검색.
    rows 리스트 그대로 반환.
    """
    rows = _get(f"/servers/{server}/characters",
                characterName=name,
                wordType=wordType,
                limit=limit).get("rows", [])
    return rows
