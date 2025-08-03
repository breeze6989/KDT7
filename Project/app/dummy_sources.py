# 카메라 8대 (id·이름·상태·위치·영상파일)
cctv_list = [
    dict(id=1, name="CCTV 1", status="정상",      lat=37.5502, lng=127.1484, video="videos/cctv1.mp4"),
    dict(id=2, name="CCTV 2", status="화재감지",  lat=37.5623, lng=126.8242, video="videos/cctv2.mp4"),
    dict(id=3, name="CCTV 3", status="오탐",      lat=37.5172, lng=127.0473, video="videos/cctv3.mp4"),
    dict(id=4, name="CCTV 4", status="정상",      lat=37.5665, lng=126.9015, video="videos/cctv4.mp4"),
    dict(id=5, name="CCTV 5", status="정상",      lat=37.4838, lng=127.0325, video="videos/cctv5.mp4"),
    dict(id=6, name="CCTV 6", status="화재감지",  lat=37.5145, lng=127.1050, video="videos/cctv6.mp4"),
    dict(id=7, name="CCTV 7", status="정상",      lat=37.5325, lng=126.9903, video="videos/cctv7.mp4"),
    dict(id=8, name="CCTV 8", status="정상",      lat=37.5635, lng=127.0365, video="videos/cctv8.mp4"),
]
# id → dict 빠른 조회용
cctv_by_id = {c["id"]: c for c in cctv_list}
