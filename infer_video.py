# 3_infer_video.py (640x640 + ROI CHUẨN + INFERENCE HOÀN HẢO)
import torch
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from pathlib import Path
import json
import sys

# ===================== SIAMESE MODEL =====================
class SiameseNetwork(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(pretrained=False)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, dim), torch.nn.BatchNorm1d(dim)
        )
    def embed(self, x):
        return self.head(self.backbone(x).flatten(1))

# ===================== TRACKER WITH ROI =====================
class Tracker:
    def __init__(self, yolo_path, siamese_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(yolo_path)
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(siamese_path, map_location=self.device))
        self.model.eval()

        # Transform giống training
        self.transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def load_refs(self, paths):
        embs = []
        for p in paths:
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = self.transform(image=img)['image'].unsqueeze(0).to(self.device)
            embs.append(self.model.embed(t))
        self.ref_emb = torch.mean(torch.stack(embs), dim=0)  # [1, 512]

    def _crop_roi(self, frame, box, scale=1.5, target_size=640):
        """Giống hệt 1_prepare_data.py"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1) * scale
        size = max(size, 64)
        rx1 = int(max(0, cx - size / 2))
        ry1 = int(max(0, cy - size / 2))
        rx2 = int(min(w, cx + size / 2))
        ry2 = int(min(h, cy + size / 2))
        if rx2 <= rx1 or ry2 <= ry1: return None
        roi = frame[ry1:ry2, rx1:rx2]
        return cv2.resize(roi, (target_size, target_size))

    def track(self, video_path, out_json, sim_thr=0.65):
        cap = cv2.VideoCapture(str(video_path))
        results = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            # YOLO detect
            dets = self.yolo(frame, conf=0.3, verbose=False)[0].boxes.xyxy.cpu().numpy()
            best_sim = 0
            best_box = None

            for box in dets:
                roi = self._crop_roi(frame, box, scale=1.5)  # ROI CHUẨN
                if roi is None: continue

                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                t = self.transform(image=roi_rgb)['image'].unsqueeze(0).to(self.device)
                sim = F.cosine_similarity(self.ref_emb, self.model.embed(t)).item()

                if sim > best_sim:
                    best_sim, best_box = sim, box

            if best_box is not None and best_sim > sim_thr:
                x1, y1, x2, y2 = map(int, best_box)
                results.append({'frame': frame_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

            frame_id += 1

        cap.release()
        sub = {
            'video_id': Path(video_path).parent.name,
            'annotations': [{'bboxes': results}]
        }
        Path(out_json).write_text(json.dumps(sub, indent=2))
        print(f"Tracked {len(results)} frames → {out_json}")

# ===================== MAIN =====================
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python 3_infer_video.py <video_folder> <yolo.pt> <siamese.pth>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    refs = sorted((folder / 'object_images').glob('*.jpg'))[:3]
    if len(refs) < 3:
        print("Cần ít nhất 3 ảnh reference!")
        sys.exit(1)

    tracker = Tracker(sys.argv[2], sys.argv[3])
    tracker.load_refs(refs)
    tracker.track(folder / 'drone_video.mp4', f"submission_{folder.name}.json")