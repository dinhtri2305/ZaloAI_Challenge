# 3_infer_video.py
import torch, torch.nn.functional as F, cv2, albumentations as A
from ultralytics import YOLO
from pathlib import Path
import json, sys

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
    def embed(self, x): return self.head(self.backbone(x).flatten(1))

class Tracker:
    def __init__(self, yolo_path, siamese_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(yolo_path)
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(siamese_path, map_location=self.device))
        self.model.eval()
        self.transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            A.pytorch.ToTensorV2()
        ])

    def load_refs(self, paths):
        embs = []
        for p in paths:
            img = cv2.imread(str(p))
            img = cv2.resize(img, (640, 640))
            t = self.transform(image=img)['image'].unsqueeze(0).to(self.device)
            embs.append(self.model.embed(t))
        self.ref_emb = torch.mean(torch.stack(embs), dim=0)

    def track(self, video_path, out_json, sim_thr=0.65):
        cap = cv2.VideoCapture(str(video_path))
        results = []
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            dets = self.yolo(frame, conf=0.3, verbose=False)[0].boxes.xyxy.cpu().numpy()
            best_sim = 0
            best_box = None
            for box in dets:
                x1,y1,x2,y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                crop = cv2.resize(crop, (640, 640))
                t = self.transform(image=crop)['image'].unsqueeze(0).to(self.device)
                sim = F.cosine_similarity(self.ref_emb, self.model.embed(t)).item()
                if sim > best_sim:
                    best_sim, best_box = sim, (x1,y1,x2,y2)
            if best_box and best_sim > sim_thr:
                x1,y1,x2,y2 = best_box
                results.append({'frame': frame_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            frame_id += 1
        cap.release()
        sub = {'video_id': Path(video_path).parent.name, 'annotations': [{'bboxes': results}]}
        Path(out_json).write_text(json.dumps(sub, indent=2))
        print(f"[3] Tracked {len(results)} frames → {out_json}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python 3_infer_video.py <video_folder> <yolo.pt> <siamese.pth>")
        sys.exit(1)
    folder = Path(sys.argv[1])
    refs = sorted((folder/'object_images').glob('*.jpg'))[:3]
    tracker = Tracker(sys.argv[2], sys.argv[3])
    tracker.load_refs(refs)
    tracker.track(folder/'drone_video.mp4', f"submission_{folder.name}.json")