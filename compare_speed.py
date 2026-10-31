"""走行速度を変えて動作時間を比較する、簡単な確認用スクリプト。"""
from pathlib import Path
from src.ai_vision import train_model
import src.robot_sim as sim

model, _ = train_model(Path(__file__).parent / 'results' / 'model.joblib')
for speed in [0.35, 0.60]:
    sim.MAX_V = speed
    _, _, result = sim.simulate(model, max_time=200)
    print('最大速度 {:.2f} m/s : {} 秒 / 正解 {} 件'.format(
        speed, result['duration_s'], result['items_correctly_sorted']))
