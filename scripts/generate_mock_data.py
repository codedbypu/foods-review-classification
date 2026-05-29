"""Generate synthetic Thai review CSV for pipeline smoke tests."""

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

data_map = {
    1: [
        "แย่มาก ไม่อร่อยเลย",
        "บริการแย่ รอนานมาก",
        "ไม่แนะนำ เนื้อเหนียวเคี้ยวไม่ออก",
        "แพงเกินไป รสชาติห่วย",
    ],
    2: [
        "พอใช้ได้ แต่ไม่ค่อยอร่อยเท่าไหร่",
        "รสชาติจืดชืด บริการค่อนข้างช้า",
        "ราคาแพงเกินคุณภาพ",
    ],
    3: [
        "รสชาติกลางๆ กินได้แก้หิว",
        "รสชาติปานกลาง ราคาถือว่าโอเค",
        "พอใช้ได้สำหรับมื้อเร่งด่วน",
    ],
    4: [
        "อาหารอร่อย บริการดีครับ",
        "รสชาติเข้มข้น ชอบมาก",
        "คุ้มค่า ต้องกลับมาซ้ำ",
    ],
    5: [
        "อร่อยมาก สุดยอดจริงๆ",
        "ดีเยี่ยม บริการประทับใจสุดๆ",
        "แนะนำมาก คุ้มค่าที่สุด",
    ],
}

stars_list = [1, 2, 3, 4, 5]
weights = [0.1, 0.1, 0.2, 0.3, 0.3]

rows = []
for _ in range(2000):
    star = random.choices(stars_list, weights=weights)[0]
    body = random.choice(data_map[star])
    rows.append({"review_body": body, "stars": star})

out_path = Path(config.MOCK_TRAIN_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Wrote {len(rows)} rows to {out_path}")
