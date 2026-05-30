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

prefixes = ["ร้านนี้", "มาทานที่นี่", "ลองแล้ว", "วันนี้", "เมื่อวาน", "เพื่อนชวน"]
suffixes = ["ครับ", "ค่ะ", "นะ", "จริงๆ", "เลย", "มาก"]
dishes = [
    "ข้าวมันไก่", "ก๋วยเตี๋ยว", "ต้มยำ", "ส้มตำ", "ข้าวผัด",
    "แกงเขียวหวาน", "สเต็ก", "เบอร์ger", "ชานม", "ขนมปัง",
]

rows = []
for i in range(2000):
    star = random.choices(stars_list, weights=weights)[0]
    core = random.choice(data_map[star])
    body = (
        f"{random.choice(prefixes)} {core} "
        f"{random.choice(dishes)} #{i % 997 + random.randint(1, 50)} "
        f"{random.choice(suffixes)}"
    )
    rows.append({"review_body": body, "stars": star})

df = pd.DataFrame(rows)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_test = max(200, int(len(df) * 0.2))
df_test = df.iloc[:n_test]
df_train = df.iloc[n_test:]

train_path = Path(config.MOCK_TRAIN_PATH)
test_path = Path(config.MOCK_TEST_PATH)
train_path.parent.mkdir(parents=True, exist_ok=True)
df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
df_test.to_csv(test_path, index=False, encoding="utf-8-sig")
print(f"Wrote train {len(df_train)} rows -> {train_path}")
print(f"Wrote test  {len(df_test)} rows -> {test_path}")
