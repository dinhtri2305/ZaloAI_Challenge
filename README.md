# Cài đặt thư mục

ZaloAI_Challenge/
├── train/ ← Dữ liệu gốc
├── data/processed/ ← Output: ảnh 640x640 + train_pairs.json
├── models/ ← .pth weights
├── 1_prepare_data.py ← Crop 640x640 RAW
├── 2_train_siamese.py ← Train Siamese 640x640
├── 3_infer_video.py ← Inference (YOLO + Siamese)
├── pipeline.py ← Chạy full pipeline
├── requirements.txt
└── README.md

# HƯỚNG DẪN CHẠY CODE:

**Bước 1:** _Cài đặt thư viện bên trong requirements.txt:_

`!pip install -r requirements.txt`

**Bước 2:** _CÁC BƯỚC TRAINING:_

2.1 `python3 prepare_data.py`

2.2 `python3 train_siamese.py full`

**Bước 3:** _TEST TRÊN 1 VIDEO:_

`python 3_infer_video.py "train/samples/Backpack_0" yolov8n.pt models/siamese_best_640.pth`
