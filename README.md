# Restaurant Sentiment And Rating Review

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

#### Option A: ติดตั้งแบบใช้ `requirements.txt` (ง่ายสุด)

> ตัวอย่างด้านล่างเป็นคำสั่งบน Windows (PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: ติดตั้งเป็นแพ็กเกจแบบ editable (แนะนำถ้าจะพัฒนาเพิ่ม)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### 3. Usage (การใช้งานผ่านสคริปต์)

สคริปต์ทั้งหมดอยู่ในโฟลเดอร์ `scripts/` และรองรับไฟล์ `.csv` หรือ `.parquet`

#### รูปแบบข้อมูลขั้นต่ำที่ต้องมี

- **คอลัมน์บังคับ**: `text`, `user_rating`
  - `text`: ข้อความรีวิว
  - `user_rating`: คะแนนดาวจากผู้ใช้ (ต้องอยู่ในช่วง 1..5)
- **คอลัมน์ทางเลือก**:
  - `lang` (ใช้แยกผล eval ตามภาษาใน `scripts/evaluate.py`)
  - `lat`, `lon` (ใช้ export GeoJSON ใน `scripts/score_and_flag.py`)

#### 3.1 Preprocess (normalize ข้อความ)

```bash
python scripts/preprocess.py --input data/raw.csv --out data/processed.parquet
```

#### 3.2 Train: Baseline (TF-IDF + XGBoost)

```bash
python scripts/train_baseline_xgb.py --input data/processed.parquet --out_dir artifacts/baseline_xgb
```

#### 3.3 Train: SOTA (Fine-tune XLM-RoBERTa 5-class)

```bash
python scripts/train_xlmr_sentiment.py --input data/processed.parquet --out_dir artifacts/xlmr
```

#### 3.4 Evaluate โมเดลที่เทรนแล้ว

```bash
python scripts/evaluate.py --input data/processed.parquet --model_type baseline_xgb --artifact_dir artifacts/baseline_xgb --out reports/baseline_metrics.json
python scripts/evaluate.py --input data/processed.parquet --model_type xlmr --artifact_dir artifacts/xlmr --out reports/xlmr_metrics.json
```

#### 3.5 Score + Integrity Check + Export สี/GeoJSON

คำสั่งนี้จะคำนวณ
- `ai_expected_rating`, `ai_pred_class`
- `delta` และ `is_anomaly` (ตรวจความต่างระหว่างดาวผู้ใช้กับ AI ด้วย threshold)
- `ai_hex_color` (แมปสีจากคะแนน)
- ไฟล์ aspects เพิ่มเติม (`*_aspects.csv|parquet`) เพื่อใช้ downstream

```bash
python scripts/score_and_flag.py --input data/processed.parquet --model_type baseline_xgb --artifact_dir artifacts/baseline_xgb --out outputs/scored.parquet
```

ถ้ามี `lat`/`lon` และอยาก export GeoJSON:

```bash
python scripts/score_and_flag.py --input data/processed.parquet --model_type xlmr --artifact_dir artifacts/xlmr --out outputs/scored.parquet --geojson_out outputs/scored.geojson
```

---

## 📈 Evaluation Results (ผลการทดสอบเบื้องต้น)

| Model | Accuracy | F1-Score (Macro) | Inference Speed (per review) | Model Size |
| --- | --- | --- | --- | --- |
| **TF-IDF + XGBoost** | --% | --% | ~ XX ms | ~ XX MB |
| **Fine-tuned XLM-RoBERTa** | --% | --% | ~ XX ms | ~ XX MB |