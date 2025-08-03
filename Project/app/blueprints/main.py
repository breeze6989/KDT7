from flask import Blueprint, render_template, request
from app.dummy_sources import cctv_list

bp = Blueprint("main", __name__)

@bp.get("/")
def dashboard():
    # ?cols=4&rows=2 식으로 배열 변경(기본 4x2)
    cols = int(request.args.get("cols", 4))
    rows = int(request.args.get("rows", 2))
    return render_template("test_03.html",
                           cctv_list=cctv_list,
                           cols=cols, rows=rows)

@bp.get("/login")
def login():
    return render_template("login.html")
