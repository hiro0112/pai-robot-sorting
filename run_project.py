"""画像分類を学習して、仕分けロボットを動かす。"""
import csv
import json
from pathlib import Path

import cv2

from src.ai_vision import train_model
from src.robot_sim import frame_to_dict, simulate


results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

# 1. 合成画像で3種類の荷物を見分けるモデルを学習
model, model_result = train_model(results_dir / 'model.joblib')

# 2. AIの判定結果を行き先として、2Dロボットを動かす
frames, camera_images, robot_result = simulate(model)

# 3. 後から動きを動画にできるように結果を保存
for number, image in camera_images.items():
    cv2.imwrite(str(results_dir / f'inspection_{number+1:02d}.png'), image)

with (results_dir / 'robot_log.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(frame_to_dict(frames[0]).keys()))
    writer.writeheader()
    for frame in frames:
        writer.writerow(frame_to_dict(frame))

with (results_dir / 'metrics.json').open('w', encoding='utf-8') as f:
    json.dump({'model': model_result, 'mission': robot_result}, f,
              indent=2, ensure_ascii=False)

print('画像の分類正解率（合成データ）: {:.1%}'.format(model_result['test_accuracy']))
print('正しく運べた荷物: {} / {}'.format(robot_result['items_correctly_sorted'],
                                    robot_result['items_total']))
print('走行終了: {} 秒'.format(robot_result['duration_s']))
if not robot_result['mission_complete']:
    print('注意: ロボットが終了位置までたどり着きませんでした')
