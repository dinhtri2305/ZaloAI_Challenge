# 1_prepare_data.py
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import shutil
import random

class DataPreparation:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / 'samples'
        self.annotations_path = self.data_dir / 'annotations' / 'annotations.json'

        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {self.annotations_path}")

        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

        self.video_to_folder = {}
        for folder in self.samples_dir.iterdir():
            if not folder.is_dir(): continue
            for ann in self.annotations:
                if ann['video_id'] == folder.name:
                    self.video_to_folder[ann['video_id']] = folder
                    break

        print(f"[1] Loaded {len(self.annotations)} videos from {self.annotations_path.parent}")

    def prepare(self, output_dir, samples_per_video=50, neg_per_ref=15, target_size=640):
        out = Path(output_dir)
        pos_dir = out / 'positives'
        neg_dir = out / 'negatives'
        ref_dir = out / 'references'
        for d in [pos_dir, neg_dir, ref_dir]: d.mkdir(parents=True, exist_ok=True)

        pairs = []
        print("[1] Extracting 640x640 RAW patches...")

        for video_data in tqdm(self.annotations):
            vid = video_data['video_id']
            bboxes = video_data['annotations'][0]['bboxes']
            folder = self.video_to_folder.get(vid)
            if not folder: continue

            video_path = folder / 'drone_video.mp4'
            ref_dir_path = folder / 'object_images'
            if not video_path.exists() or not ref_dir_path.exists(): continue

            # === Copy 3 references (640x640 RAW) ===
            refs = sorted(ref_dir_path.glob('*.jpg'))[:3]
            if len(refs) < 3: continue
            for i, src in enumerate(refs):
                img = cv2.imread(str(src))
                img = cv2.resize(img, (target_size, target_size))
                dst = ref_dir / f"{vid}_{i+1}.jpg"
                cv2.imwrite(str(dst), img)

            # === Extract positive ROIs (640x640 RAW) ===
            cap = cv2.VideoCapture(str(video_path))
            frame_ids = sorted([b['frame'] for b in bboxes])
            sampled = np.linspace(0, len(frame_ids)-1, min(samples_per_video, len(frame_ids)), dtype=int)

            for idx in sampled:
                f_id = frame_ids[idx]
                bbox = next(b for b in bboxes if b['frame'] == f_id)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_id)
                ret, frame = cap.read()
                if not ret: continue

                roi = self._crop_and_resize(frame, bbox, target_size)
                if roi is None: continue

                roi_name = f"{vid}_f{f_id}.jpg"
                cv2.imwrite(str(pos_dir / roi_name), roi)

                for i in range(1, 4):
                    pairs.append({
                        'reference': f"{vid}_{i}.jpg",
                        'sample': roi_name,
                        'label': 1,
                        'video_id': vid
                    })
            cap.release()

        # === Negative pairs ===
        print(f"[1] Generating {neg_per_ref} negatives/ref...")
        video_ids = list({p['video_id'] for p in pairs if p['label'] == 1})
        for ref_file in ref_dir.glob('*.jpg'):
            ref_vid = '_'.join(ref_file.stem.split('_')[:-1])
            others = [v for v in video_ids if v != ref_vid]
            if not others: continue
            for ovid in random.sample(others, min(neg_per_ref, len(others))):
                crops = list(pos_dir.glob(f"{ovid}_f*.jpg"))
                if not crops: continue
                crop = random.choice(crops)
                neg_path = neg_dir / crop.name
                shutil.copy(crop, neg_path)
                pairs.append({
                    'reference': ref_file.name,
                    'sample': crop.name,
                    'label': 0,
                    'video_id': ref_vid
                })

        # Save
        (out / 'train_pairs.json').write_text(json.dumps(pairs, indent=2))
        print(f"[1] DONE! {len(pairs)} pairs → {out}")

    def _crop_and_resize(self, frame, bbox, target_size=640, scale=1.5):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
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

if __name__ == "__main__":
    prep = DataPreparation('train')
    prep.prepare('data/processed', target_size=640)