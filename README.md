# Restaurant Sentiment And Rating Review

> คู่มือเชิงลึกฉบับเต็ม: **[README2.md](README2.md)** — โดยเฉพาะ **[§6 คู่มือการรัน (Runbook)](README2.md#6-คู่มือการรันแบบละเอียด-runbook)** ครอบคลุมทุกเคส (smoke, full train, bad_malloc, scratch_dir, notebook, XLM-R, production)

ระบบวิเคราะห์และตรวจสอบความน่าเชื่อถือของธุรกิจร้านอาหารด้วย AI โดยเปลี่ยนข้อความรีวิวให้กลายเป็น **"คะแนนรีวิว"**  และ Map กับสี เพื่อนำเสนอผ่านแอปพลิเคชันแผนที่อารมณ์เรืองแสง (Neon Sentiment Heatmap) พร้อมระบบตรวจสอบความสอดคล้องของรีวิวเพื่อตรวจจับรีวิวปลอมหรือความผิดปกติของข้อมูล

---

## 🌟 Key Features (ฟีเจอร์เด่นของระบบ)

1. **Multilingual Aspect-Based Sentiment Analysis (ABSA):** สกัดระดับความรู้สึก (1-5 ดาว) แยกตามมิติสำคัญ เช่น รสชาติอาหาร (Food), การบริการ (Service), ราคา (Price) และภาพรวม (General)
2. **Sentiment-to-Color Mapping:** แปลงผลลัพธ์คะแนนความรู้สึกเป็นรหัสสีตามจิตวิทยา เช่น 🟦 5 ดาว (ดีเยี่ยม), 🟩 4 ดาว (ดี), 🟨 3 ดาว (ปานกลาง), 🟧 2 ดาว (ไม่ค่อยดี), 🟥 1 ดาว (แย่มาก)
3. **Integrity & Anomaly Detection (ระบบจับผิด):** เปรียบเทียบคะแนนที่ AI ประเมินได้จากข้อความ กับ คะแนนดาวที่ผู้ใช้กดจริง หากมีความต่างกันอย่างรุนแรง ($\Delta \ge 2.0$) ระบบจะแจ้งเตือนว่าเป็นรีวิวต้องสงสัยทันที

---

## 🧠 AI Models & Benchmarking (สถาปัตยกรรมโมเดลและการวัดผล)

โปรเจกต์นี้ทำการเปรียบเทียบประสิทธิภาพ (Benchmarking) ระหว่าง 2 โมเดล เพื่อหาจุดสมดุลระหว่างความแม่นยำและความเร็วในการประมวลผล (Accuracy vs Speed Trade-off):

* **State-of-the-Art Model (Fine-tuned XLM-RoBERTa):** โมเดลกลุ่ม Transformer แบบ Multilingual เรียนรู้ความรู้สึกข้ามสายภาษา (Cross-lingual Transfer) โดยทดลองพัฒนาด้วยเทคนิค Transfer Learning จากชุดข้อมูลขนาดใหญ่อย่าง *Amazon Fine Food Dataset* สู่ชุดข้อมูลรีวิวภาษาไทย
* **Baseline Model (TF-IDF + XGBoost):** โมเดลสายสถิติตระกูล Tree-based โดยใช้ `PyThaiNLP` เป็น Custom Tokenizer ในการตัดคำผสมสองภาษา (ไทย-อังกฤษ) เพื่อสร้างตารางฟีเจอร์ความถี่คำ 

---

## 📊 Dataset Used & Spatial Splitting (ชุดข้อมูลและการแบ่งสัดส่วน)

โปรเจกต์นี้ใช้ชุดข้อมูลผสมผสานสองภาษา (Multilingual Datasets) รวมทั้งสิ้นกว่า 500,000 ข้อความ โดยแบ่งโครงสร้างอย่างเป็นระบบเพื่อใช้ในการเทรน (Training) และทดสอบความคงเส้นคงวาของโมเดล (Independent Evaluation) ดังนี้:

### 🎯 1. Training & Validation Datasets (สัดส่วน Train 70% / Validation 30%)
ชุดข้อมูลหลักที่เปิดให้ทั้งโมเดล **XLM-RoBERTa** และ **TF-IDF + XGBoost** ได้เรียนรู้โครงสร้างของภาษาและอารมณ์ความรู้สึก:

* **🇺🇸 English Language Domain:**
    * [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (~400,000 รีวิว) — ชุดข้อมูลขนาดใหญ่สำหรับสร้างฐานความรู้ (Base Knowledge) ด้าน Sentiment Analysis
* **🇹🇭 Thai Language Domain:**
    * [Wongnai Corpus](https://github.com/wongnai/wongnai-corpus) (~40,000 รีวิว) — คลังข้อมูลรีวิวร้านอาหารภาษาไทยระดับมาตรฐาน
    * [iamwarint/wongnai-restaurant-review](https://huggingface.co/datasets/iamwarint/wongnai-restaurant-review) (~20,000 รีวิว) — ชุดข้อมูลรีวิวภาษาไทยเพิ่มเติมเพื่อเพิ่มความหลากหลายของคำศัพท์

### 🧪 2. Independent Evaluation Datasets (Testing Only 100%)
ชุดข้อมูลทดสอบภายนอก (Holdout Sets) ที่โมเดลไม่เคยเห็นในขั้นตอนการเทรน เพื่อใช้วัดประสิทธิภาพที่แท้จริงและใช้ประมวลผลร่วมกับระบบตรวจจับรีวิวปลอม (Anomaly Detection):

* **🇹🇭 Thai Language Testing:**
    * [wttw/restaurant_review](https://huggingface.co/datasets/wttw/restaurant_review) (~71,000 รีวิว) — ชุดข้อมูลขนาดใหญ่สำหรับทดสอบความแม่นยำและความเร็วของโมเดลในโดเมนร้านอาหารไทย
* **🇺🇸 English Language Testing:**
    * [joebeachcapital/restaurant-reviews](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews) (~10,000 รีวิว) — ชุดข้อมูลสำหรับทดสอบประสิทธิภาพโมเดลในโดเมนร้านอาหารฝั่งภาษาอังกฤษ

---

## 🛠️ Tech Stack (เครื่องมือที่ใช้พัฒนา)

* **Language:** Python 3.x
* **NLP & Deep Learning:** Hugging Face Transformers, PyTorch / TensorFlow
* **Machine Learning:** Scikit-learn, XGBoost
* **Thai NLP:** PyThaiNLP (newmm engine)
* **Data Manipulation:** Pandas, NumPy

---

## 🚀 Getting Started (วิธีติดตั้งและใช้งาน)

### 1. Clone the Repository
```bash
git clone https://github.com/codedbypu/foods-review-classification.git
cd foods-review-classification
```

### 2. Install Dependencies (แนะนำให้ใช้ Virtual Environment)

> ตัวอย่างด้านล่างเป็นคำสั่งบน Windows (PowerShell) — Linux/macOS ใช้ `source .venv/bin/activate` แทน

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**ให้ import `rris` ได้** — เลือกอย่างใดอย่างหนึ่ง:

| วิธี | เมื่อไหร่ใช้ |
| --- | --- |
| `pip install -e .` | แนะนำสำหรับพัฒนา / Jupyter / รันสคริปต์ซ้ำๆ |
| ไม่ต้องติดตั้งแพ็กเกจ | สคริปต์ใน `scripts/` โหลด `scripts/_bootstrap.py` ให้เพิ่ม `src/` เข้า `sys.path` อัตโนมัติ |

```bash
pip install -e .
```

ดู pipeline แบบครบวงจรได้ที่ notebook รากโปรเจกต์: [`food-review.ipynb`](food-review.ipynb)

### 3. Usage (การใช้งานผ่านสคริปต์)

รันจาก **root ของ repo** (`foods-review-classification/`). สคริปต์อยู่ใน `scripts/` รองรับ `.csv` / `.parquet` และใช้ `rris.data.io.read_reviews` แปลง schema ให้

สคริปต์ทุกตัวรองรับ `--no_progress` เพื่อปิด tqdm progress bar

**แก้ error / kernel crash:** ดู [docs/ERRORS_AND_FIXES.md](docs/ERRORS_AND_FIXES.md)  
**ผลรันเช็คล่าสุด:** [docs/RUNTIME_CHECK_REPORT.md](docs/RUNTIME_CHECK_REPORT.md) — แต่ละสคริปต์ใน `scripts/` และโมดูล runtime ใน `src/rris/` มีบล็อก `COMMON ERRORS` ที่หัวไฟล์

#### รูปแบบข้อมูล

- **มาตรฐาน (หลัง preprocess / parquet)**: `text`, `user_rating` (1..5)
- **Wongnai CSV โดยตรง** (เช่น Hugging Face export): คอลัมน์ `review_body`, `stars` — `read_reviews` แมปเป็น `text`, `user_rating` อัตโนมัติ
- **ทางเลือก**: `lang` (แยกผล eval), `lat`, `lon` (export GeoJSON ใน `score_and_flag.py`)

ตัวอย่างไฟล์ใน repo: `data/wongnai-restaurant-review_train.csv`

#### 3.1 Preprocess (normalize ข้อความ)

```bash
python scripts/preprocess.py --input data/wongnai-restaurant-review_train.csv --out data/wongnai_processed.parquet
```

| Flag | คำอธิบาย |
| --- | --- |
| `--input` | ไฟล์ `.csv` / `.parquet` ต้นทาง |
| `--out` | ไฟล์ผลลัพธ์หลัง normalize |
| `--no_progress` | ปิด progress bar |

#### 3.2 Train: Baseline (TF-IDF + XGBoost)

```bash
python scripts/train_baseline_xgb.py ^
  --input data/wongnai_processed.parquet ^
  --out_dir artifacts/baseline_xgb
```

อ่าน Wongnai CSV ได้โดยตรง (ไม่ต้อง preprocess ก่อน ถ้ายอมรับ normalize ตอน train):

```bash
python scripts/train_baseline_xgb.py --input data/wongnai-restaurant-review_train.csv --out_dir artifacts/baseline_xgb
```

| Flag | Default | คำอธิบาย |
| --- | --- | --- |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` — **auto** ใช้ CUDA ถ้า `torch.cuda.is_available()` ไม่งั้น CPU + `--n_jobs` |
| `--n_jobs` | `-1` | worker สำหรับตัดคำ / TF-IDF / XGBoost บน CPU |
| `--max_rows` | (ไม่จำกัด) | จำกัดแถวสำหรับ smoke / demo |
| `--scratch_dir` | (ไม่มี) | โฟลเดอร์ temp บนไดรฟ์ที่ว่าง (`RRIS_SCRATCH_DIR`) — ดู [README2 §4.4](README2.md#44-เลือกไดรฟ์--โฟลเดอร์-temp---scratch_dir) |
| `--no_progress` | off | ปิด progress bar |

**GPU (XGBoost):** `--device auto` ตรวจ NVIDIA CUDA ผ่าน PyTorch แล้วตั้ง `device=cuda` ให้ XGBoost — ถ้าไม่มี GPU จะ fallback CPU หลาย thread อัตโนมัติ (Intel iGPU **ไม่** เร่ง XGBoost)

ถ้า VRAM เต็มหรือ CUDA error ให้บังคับ CPU:

```bash
python scripts/train_baseline_xgb.py --input data/wongnai_processed.parquet --out_dir artifacts/baseline_xgb --device cpu --n_jobs -1
```

Smoke test สั้นๆ:

```bash
python scripts/train_baseline_xgb.py --input data/wongnai-restaurant-review_train.csv --out_dir artifacts/baseline_smoke --max_rows 2000 --device auto
```

#### 3.3 Train: SOTA (Fine-tune XLM-RoBERTa 5-class)

```bash
python scripts/train_xlmr_sentiment.py --input data/wongnai_processed.parquet --out_dir artifacts/xlmr
```

| Flag | หมายเหตุ |
| --- | --- |
| `--no_progress` | ปิด tqdm ระหว่างเทรน |
| (อื่นๆ) | `--model_name`, `--epochs`, `--train_batch_size`, … ดู `--help` |

> **CUDA:** XLM-R ใช้ PyTorch — แนะนำ GPU (VRAM หลาย GB) สคริปต์ log `Torch CUDA available: True/False` ตอนเริ่ม การเทรนเต็มชุด Wongnai ใช้เวลานาน; ทดลองกับ subset หรือ baseline ก่อน

#### 3.4 Evaluate โมเดลที่เทรนแล้ว

```bash
python scripts/evaluate.py --input data/wongnai_processed.parquet --model_type baseline_xgb --artifact_dir artifacts/baseline_xgb --out reports/baseline_metrics.json
python scripts/evaluate.py --input data/wongnai_processed.parquet --model_type xlmr --artifact_dir artifacts/xlmr --out reports/xlmr_metrics.json --batch_size 32
```

| Flag | Default | คำอธิบาย |
| --- | --- | --- |
| `--batch_size` | `32` | batch สำหรับ inference XLM-R |
| `--n_jobs` | `-1` | parallel สำหรับ baseline vectorize |
| `--no_progress` | off | ปิด progress bar |

#### 3.5 Score + Integrity Check + Export สี/GeoJSON

คำนวณ `ai_expected_rating`, `ai_pred_class`, `delta`, `is_anomaly`, `ai_hex_color` และไฟล์ aspects (`*_aspects.csv|parquet`)

```bash
python scripts/score_and_flag.py --input data/wongnai_processed.parquet --model_type baseline_xgb --artifact_dir artifacts/baseline_xgb --out outputs/scored.parquet
```

| Flag | Default | คำอธิบาย |
| --- | --- | --- |
| `--batch_size` | `32` | batch สำหรับ XLM-R scoring |
| `--n_jobs` | `-1` | parallel สำหรับ baseline |
| `--no_progress` | off | ปิด progress bar |
| `--geojson_out` | (ไม่มี) | export GeoJSON ถ้ามี `lat`, `lon` |

```bash
python scripts/score_and_flag.py --input data/wongnai_processed.parquet --model_type baseline_xgb --artifact_dir artifacts/baseline_xgb --out outputs/scored.parquet --batch_size 32 --n_jobs -1
```

---

## 📈 Evaluation Results (ผลการทดสอบเบื้องต้น)

| Model | Accuracy | F1-Score (Macro) | Inference Speed (per review) | Model Size |
| --- | --- | --- | --- | --- |
| **TF-IDF + XGBoost** | --% | --% | ~ XX ms | ~ XX MB |
| **Fine-tuned XLM-RoBERTa** | --% | --% | ~ XX ms | ~ XX MB |