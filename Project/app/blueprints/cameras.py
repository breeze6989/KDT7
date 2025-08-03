from flask import Blueprint, Response, abort
from app.dummy_sources import cctv_by_id
from ..services.streamer import thumbnail_frame, mjpeg_stream

bp = Blueprint("cameras", __name__)

@bp.get("/<int:cid>/thumbnail")
def thumbnail(cid):
    cam = cctv_by_id.get(cid)
    if not cam:
        abort(404)
    jpg = thumbnail_frame(cam["video"])
    if jpg is None:
        abort(500)
    return Response(jpg, mimetype="image/jpeg")

@bp.get("/<int:cid>/stream")
def stream(cid):
    cam = cctv_by_id.get(cid)
    if not cam:
        abort(404)
    return Response(mjpeg_stream(cam["video"]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
