# HƯỚNG DẪN CHẠY CODE:

**Bước 1:** _Cài đặt thư viện bên trong requirements.txt:_
`!pip install -r requirements.txt`
**Bước 2:** _cÁC BƯỚC TRAINING:_
2.1 `python3 prepare_data.py`
2.2 `python3 train_siamese.py full`
**Bước 3:** _TEST TRÊN 1 VIDEO:_
`python 3_infer_video.py "train/samples/Backpack_0" yolov8n.pt models/siamese_best_640.pth`
