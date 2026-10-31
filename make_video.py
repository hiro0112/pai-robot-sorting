"""走行ログから、車体と2つの車輪が見える動画を作る。"""
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / 'results'
WIDTH, HEIGHT = 960, 540
FPS = 10

# OpenCVはBGRの順で色を指定
DARK = (54, 58, 66)
LIGHT = (250, 250, 249)
RED = (84, 99, 219)
BLUE = (211, 145, 85)
GREEN = (112, 170, 97)
COLORS = {'MEDICINE': RED, 'WATER': BLUE, 'FOOD': GREEN}
DESTINATIONS = [('MEDICINE', 1.80), ('WATER', 0.0), ('FOOD', -1.80)]


def xy(x, y):
    """シミュレーション上のメートルを画像上のピクセルへ変換。"""
    return int(358 + float(x) * 90), int(277 - float(y) * 90)


def text(img, message, x, y, scale=0.48, color=DARK, thickness=1):
    cv2.putText(img, str(message), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def robot_points(x, y, angle, coords):
    """ロボットから見た座標をワールド座標へ回転させる。前方がローカルx方向。"""
    cs, sn = math.cos(angle), math.sin(angle)
    return np.array([xy(x + a*cs - b*sn, y + a*sn + b*cs)
                     for a, b in coords], dtype=np.int32)


def polygon(img, x, y, angle, corners, color, outline=None, width=1):
    points = robot_points(x, y, angle, corners)
    cv2.fillConvexPoly(img, points, color, lineType=cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(img, [points], True, outline, width, cv2.LINE_AA)


def draw_package(img, px, py, label, scale=1):
    color = COLORS[label]
    size = int(12*scale)
    cv2.rectangle(img, (px-size, py-size), (px+size, py+size), color, -1)
    cv2.rectangle(img, (px-size, py-size), (px+size, py+size), (72,72,72), 1)
    cv2.line(img, (px-size+3, py), (px+size-3, py), (240,240,240), 2)
    cv2.line(img, (px,py-size+3), (px,py+size-3), (240,240,240), 1)


def draw_robot(img, row):
    """簡単な差動二輪ロボット。車体・左右の車輪・荷台・カメラを描く。"""
    x, y, a = row['x'], row['y'], row['yaw']
    # 走行軌跡の上に2つの黒い駆動輪を描く。車輪間隔はモデルと同じ0.20 m。
    for center_y in (-0.105, 0.105):
        polygon(img, x, y, a,
                [(-0.17, center_y-0.048), (0.12, center_y-0.048),
                 (0.12, center_y+0.048), (-0.17, center_y+0.048)],
                (38, 43, 46), (15, 18, 20), 1)
        # 車輪の回転方向がわかる目印
        t = row['t'] * (row['right_wheel_rad_s'] if center_y > 0 else row['left_wheel_rad_s'])
        cx, cy = robot_points(x, y, a, [(-0.025, center_y)])[0]
        offset = int(5*math.sin(t))
        cv2.line(img, (cx-3,cy+offset), (cx+3,cy+offset), (198, 198, 197), 1)

    # 車体
    polygon(img, x, y, a,
            [(-0.22,-0.092),(0.19,-0.092),(0.19,0.092),(-0.22,0.092)],
            (174, 174, 174), (73, 78, 81), 2)
    polygon(img, x, y, a,
            [(-0.14,-0.066),(0.12,-0.066),(0.12,0.066),(-0.14,0.066)],
            (206, 206, 206), (135, 135, 135), 1)

    # 前方に2本のフォーク（荷物を載せる表現。物理把持はしていない）。
    for ly in (-0.072, 0.072):
        points=robot_points(x,y,a,[(0.17,ly),(0.40,ly)])
        cv2.line(img,tuple(points[0]),tuple(points[1]),(83, 91, 93),3,cv2.LINE_AA)

    # 小さなカメラ（画像を推論していることを示す）
    cx, cy = robot_points(x,y,a,[(0.11,0.0)])[0]
    cv2.circle(img,(cx,cy),5,(44,85,121),-1,cv2.LINE_AA)
    cv2.circle(img,(cx,cy),2,(218,233,240),-1,cv2.LINE_AA)

    # ロボットの前後方向を示す小さな線
    p1,p2=robot_points(x,y,a,[(0.13,0.0),(0.26,0.0)])
    cv2.arrowedLine(img,tuple(p1),tuple(p2),(43,70,90),2,cv2.LINE_AA,tipLength=0.4)

    # 搬送中の箱はフォークに載る
    if row['carried'] and row['detected'] in COLORS:
        px,py=robot_points(x,y,a,[(0.30,0.0)])[0]
        draw_package(img,px,py,row['detected'],0.75)


def draw_frame(row, rows, scans, index, total):
    img=np.full((HEIGHT,WIDTH,3), LIGHT, np.uint8)
    text(img,'AI sorting robot (2D simulation)',32,36,.72,DARK,2)
    text(img,'actual speed: 1x',800,35,.43)
    cv2.rectangle(img,(32,58),(713,490),(215,218,219),1)

    # 走行エリア
    cv2.rectangle(img,xy(-3.07,2.32),xy(3.07,-2.32),(245,246,247),-1)
    for xx in range(-3,4):
        cv2.line(img,xy(xx,-2.3),xy(xx,2.3),(231,233,233),1)
    for yy in range(-2,3):
        cv2.line(img,xy(-3.0,yy),xy(3.0,yy),(231,233,233),1)

    # 出発地点と未搬送の荷物
    pickup=xy(-2.55,0)
    cv2.rectangle(img,(pickup[0]-29,pickup[1]-31),(pickup[0]+29,pickup[1]+31),(224,233,237),-1)
    cv2.rectangle(img,(pickup[0]-29,pickup[1]-31),(pickup[0]+29,pickup[1]+31),(145,160,167),2)
    text(img,'PICK UP',pickup[0]-35,pickup[1]-44,.41)
    if row['phase'] in ('TO_PICKUP','INSPECT') and row['parcel_index'] <= total:
        label=row['ground_truth']
        draw_package(img,pickup[0],pickup[1],label,.78)

    # 搬送先（3種類の箱）
    for name, y in DESTINATIONS:
        px,py=xy(2.30,y)
        cv2.rectangle(img,(px-39,py-27),(px+39,py+27),(237,239,239),-1)
        cv2.rectangle(img,(px-39,py-27),(px+39,py+27),COLORS[name],3)
        text(img,name[:3],px-24,py+7,.55,COLORS[name],2)

    # 走行した道筋
    trail=np.array([xy(d['x'],d['y']) for d in rows[:index+1:6]],np.int32)
    if len(trail)>1:
        cv2.polylines(img,[trail],False,(178,178,173),2,cv2.LINE_AA)

    # 分類中はセンサが向いていることを描画
    if row['phase']=='INSPECT':
        ox,oy=robot_points(row['x'],row['y'],row['yaw'],[(0.20,0)])[0]
        cv2.circle(img,(ox,oy),17,(197,222,239),2,cv2.LINE_AA)

    draw_robot(img,row)
    # 右側：最低限の動作ログ
    cv2.line(img,(727,59),(727,490),(221,224,224),1)
    text(img,'Elapsed: %.1f s'%row['t'],747,95,.53)
    text(img,'Completed: %d / %d'%(row['delivered'],total),747,130,.53)
    status={'TO_PICKUP':'Go to pickup','INSPECT':'Classifying...',
            'TO_BIN':'Carrying a box','UNLOAD':'Unloading','COMPLETED':'Finished'}
    text(img,'Robot state',747,182,.48)
    text(img,status.get(row['phase'],row['phase']),747,207,.48,(55,99,122),2)
    text(img,'Camera / MLP',747,259,.48)
    if row['detected'] in COLORS:
        text(img,'Result: '+row['detected'],747,283,.45,COLORS[row['detected']],2)
        image=scans.get(row['parcel_index'])
        if image is not None:
            cv2.rectangle(img,(744,297),(850,403),(180,180,180),1)
            img[300:400,747:847] = cv2.resize(image,(100,100))
    else:
        text(img,'waiting for image',747,283,.40)
    text(img,'Wheel speeds (rad/s)',747,437,.42)
    text(img,'L: %+5.1f   R: %+5.1f'%(row['left_wheel_rad_s'],row['right_wheel_rad_s']),747,461,.43)
    text(img,'AI classification -> destination -> wheel motion',35,518,.43)
    text(img,'Virtual loading; no real robot',675,518,.36)
    return img


def main():
    with (OUT/'robot_log.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    for d in rows:
        for key in ['t','x','y','yaw','left_wheel_rad_s','right_wheel_rad_s']:
            d[key]=float(d[key])
        for key in ['parcel_index','delivered']:
            d[key]=int(d[key])
        d['carried']=(d['carried']=='True')
    count=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))['mission']['items_total']
    scans={i:cv2.imread(str(OUT / ('inspection_%02d.png' % i))) for i in range(1,count+1)}

    temp=OUT/'video_temp.mp4'
    video=OUT/'sorting_robot.mp4'
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(WIDTH,HEIGHT))
    if not writer.isOpened():
        raise RuntimeError('MP4 writer could not open')
    for i in range(0,len(rows),2):
        writer.write(draw_frame(rows[i],rows,scans,i,count))
    writer.release()
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(temp),
                    '-c:v','libx264','-crf','24','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(video)],check=True)
    temp.unlink()
    print('saved:',video)


if __name__=='__main__':
    main()
