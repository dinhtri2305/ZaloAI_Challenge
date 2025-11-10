# # 2_train_siamese.py
# import torch, torch.nn as nn, torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import cv2, albumentations as A
# from pathlib import Path
# import json
# from tqdm import tqdm
# import torchvision.models as models

# class SiameseNetwork(nn.Module):
#     def __init__(self, dim=512):
#         super().__init__()
#         resnet = models.resnet50(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])
#         self.head = nn.Sequential(
#             nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(1024, dim), nn.BatchNorm1d(dim)
#         )
#         self.sim = nn.Sequential(
#             nn.Linear(dim*2, 256), nn.ReLU(),
#             nn.Linear(256, 1), nn.Sigmoid()
#         )
#     def forward_one(self, x): return self.head(self.backbone(x).flatten(1))
#     def forward(self, a, b): return self.sim(torch.cat([self.forward_one(a), self.forward_one(b)], 1))

# class SiameseDataset(Dataset):
#     def __init__(self, root, json_file):
#         self.root = Path(root)
#         self.pairs = json.loads(Path(json_file).read_text())
#         self.transform = self._get_transform()
#     def _get_transform(self):
#         return A.Compose([
#             # === PREPROCESSING ===
#             A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.8),
#             A.GaussNoise(var_limit=(10, 50), p=0.3),

#             # === AUGMENTATION ===
#             A.Rotate(limit=180, p=0.8),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#             A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

#             # === YOLO-STYLE ===
#             A.OneOf([
#                 A.Mosaic(p=1.0),
#                 A.MixUp(p=1.0),
#             ], p=0.5),

#             # === MULTI-SCALE ===
#             A.RandomScale(scale_limit=0.5, p=0.5),
#             A.PadIfNeeded(640, 640, border_mode=cv2.BORDER_CONSTANT),

#             # === FINAL ===
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             A.pytorch.ToTensorV2()
#         ], p=1.0)

#     def __len__(self): return len(self.pairs)
#     def __getitem__(self, i):
#         p = self.pairs[i]
#         ref = cv2.imread(str(self.root/'references'/p['reference']))
#         sample_dir = 'positives' if p['label'] == 1 else 'negatives'
#         sample = cv2.imread(str(self.root/sample_dir/p['sample']))
#         t = self.transform(image=ref, image2=sample)
#         return t['image'], t['image2'], torch.tensor([p['label']], dtype=torch.float32)

# # def train():
# #     dataset = SiameseDataset('data/processed', 'data/processed/train_pairs.json')
# #     loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     model = SiameseNetwork().to(device)
# #     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# #     crit = nn.BCELoss()

# #     print(f"[2] Training on {device}")
# #     for epoch in range(40):
# #         model.train()
# #         loss_sum = correct = total = 0
# #         for ref, sample, label in tqdm(loader, desc=f"Epoch {epoch+1}"):
# #             ref, sample, label = ref.to(device), sample.to(device), label.to(device)
# #             opt.zero_grad()
# #             pred = model(ref, sample)
# #             loss = crit(pred, label)
# #             loss.backward()
# #             opt.step()
# #             loss_sum += loss.item()
# #             correct += ((pred > 0.5) == label).sum().item()
# #             total += label.size(0)
# #         print(f"   Loss: {loss_sum/len(loader):.4f} | Acc: {correct/total:.4f}")
# #         if (epoch+1) % 10 == 0:
# #             torch.save(model.state_dict(), f"models/siamese_epoch{epoch+1}.pth")
# #     torch.save(model.state_dict(), "models/siamese_final.pth")
# #     print("[2] DONE!")
# def train():
#     dataset = SiameseDataset('data/processed', 'data/processed/train_pairs.json')
#     loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # batch nhỏ
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = SiameseNetwork().to(device)
#     crit = nn.BCELoss()
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#     print(f"[TEST MODE] Running 2 batches only...")
#     model.train()
#     for i, (ref, sample, label) in enumerate(loader):
#         if i >= 2:  # DỪNG SAU 2 BATCH
#             print(f"Stopped after {i} batches.")
#             break
            
#         ref, sample, label = ref.to(device), sample.to(device), label.to(device)
#         opt.zero_grad()
#         pred = model(ref, sample)
#         loss = crit(pred, label)
#         loss.backward()
#         opt.step()
        
#         print(f"   Batch {i+1}: Loss = {loss.item():.4f} | Pred = {pred.squeeze().tolist()}")

#     print("[TEST] Pipeline chạy OK! Có thể train full.")
# if __name__ == "__main__":
#     Path("models").mkdir(exist_ok=True)
#     train()
# 2_train_siamese.py (ĐÃ SỬA LỖI)
# 2_train_siamese.py (640x640 - HOÀN CHỈNH, KHÔNG LỖI)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
from tqdm import tqdm
import torchvision.models as models
import sys

# ===================== MODEL =====================
class SiameseNetwork(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.head = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, dim), nn.BatchNorm1d(dim)
        )
        self.sim = nn.Sequential(
            nn.Linear(dim*2, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.backbone(x).flatten(1)
        return self.head(x)

    def forward(self, a, b):
        ea = self.forward_one(a)
        eb = self.forward_one(b)
        return self.sim(torch.cat([ea, eb], dim=1))

# ===================== DATASET =====================
class SiameseDataset(Dataset):
    def __init__(self, root, json_file):
        self.root = Path(root)
        with open(json_file, 'r') as f:
            self.pairs = json.load(f)
        self.transform = self._get_transform()
        print(f"Loaded {len(self.pairs)} pairs")

    def _get_transform(self):
        return A.Compose([
            # === PREPROCESSING ===
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8,8), p=0.8),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # === AUGMENTATION ===
            A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

            # === MULTI-SCALE + RESIZE 640x640 ===
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.Resize(640, 640),  # BẮT BUỘC ĐỒNG NHẤT

            # === CUTOUT (thay Mosaic/Mixup) ===
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=64, max_width=64, fill_value=0, p=1.0),
                A.GridDropout(ratio=0.4, p=1.0),
            ], p=0.5),

            # === NORMALIZE + ToTensor ===
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        try:
            item = self.pairs[idx]
            ref_path = self.root / 'references' / item['reference']
            sample_dir = 'positives' if item['label'] == 1 else 'negatives'
            sample_path = self.root / sample_dir / item['sample']

            if not ref_path.exists() or not sample_path.exists():
                raise FileNotFoundError(f"Missing: {ref_path} or {sample_path}")

            ref = cv2.imread(str(ref_path))
            sample = cv2.imread(str(sample_path))
            if ref is None or sample is None:
                raise ValueError("Cannot read image")

            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

            # Transform riêng → đảm bảo 3x640x640
            ref_t = self.transform(image=ref)['image']
            sample_t = self.transform(image=sample)['image']

            return ref_t, sample_t, torch.tensor([item['label']], dtype=torch.float32)

        except Exception as e:
            print(f"Error item {idx}: {e}")
            dummy = torch.zeros(3, 640, 640)
            return dummy, dummy, torch.tensor([0.0])

# ===================== TEST MODE =====================
def train_test():
    print("SIAMESE 640x640 - TEST MODE (2 BATCHES)")
    data_dir = Path('data/processed')
    json_path = data_dir / 'train_pairs.json'

    if not json_path.exists():
        print("Run 1_prepare_data.py first!")
        return

    dataset = SiameseDataset(data_dir, json_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()

    model.train()
    for i, (a, b, y) in enumerate(loader):
        if i >= 2: break
        a, b, y = a.to(device), b.to(device), y.to(device)
        opt.zero_grad()
        p = model(a, b)
        loss = crit(p, y)
        loss.backward()
        opt.step()
        print(f"   Batch {i+1} | Loss: {loss.item():.4f} | Pred: {p.squeeze().tolist()}")

    print("TEST 640x640 THÀNH CÔNG!")

# ===================== FULL TRAIN =====================
def train_full():
    print("SIAMESE 640x640 - FULL TRAINING")
    dataset = SiameseDataset('data/processed', 'data/processed/train_pairs.json')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    crit = nn.BCELoss()

    best_loss = float('inf')
    for epoch in range(50):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for a, b, y in pbar:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            p = model(a, b)
            loss = crit(p, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/siamese_best_640.pth')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/siamese_640_epoch{epoch+1}.pth')

    torch.save(model.state_dict(), 'models/siamese_final_640.pth')
    print("FULL TRAINING 640x640 HOÀN TẤT!")

# ===================== RUN =====================
if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        train_full()
    else:
        train_test()