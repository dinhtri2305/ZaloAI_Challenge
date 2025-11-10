# 2_train_siamese.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2, albumentations as A
from pathlib import Path
import json
from tqdm import tqdm
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, dim), nn.BatchNorm1d(dim)
        )
        self.sim = nn.Sequential(
            nn.Linear(dim*2, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward_one(self, x): return self.head(self.backbone(x).flatten(1))
    def forward(self, a, b): return self.sim(torch.cat([self.forward_one(a), self.forward_one(b)], 1))

class SiameseDataset(Dataset):
    def __init__(self, root, json_file):
        self.root = Path(root)
        self.pairs = json.loads(Path(json_file).read_text())
        self.transform = self._get_transform()
    def _get_transform(self):
        return A.Compose([
            # === PREPROCESSING ===
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.8),
            A.GaussNoise(var_limit=(10, 50), p=0.3),

            # === AUGMENTATION ===
            A.Rotate(limit=180, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

            # === YOLO-STYLE ===
            A.OneOf([
                A.Mosaic(p=1.0),
                A.MixUp(p=1.0),
            ], p=0.5),

            # === MULTI-SCALE ===
            A.RandomScale(scale_limit=0.5, p=0.5),
            A.PadIfNeeded(640, 640, border_mode=cv2.BORDER_CONSTANT),

            # === FINAL ===
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ], p=1.0)

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        ref = cv2.imread(str(self.root/'references'/p['reference']))
        sample_dir = 'positives' if p['label'] == 1 else 'negatives'
        sample = cv2.imread(str(self.root/sample_dir/p['sample']))
        t = self.transform(image=ref, image2=sample)
        return t['image'], t['image2'], torch.tensor([p['label']], dtype=torch.float32)

def train():
    dataset = SiameseDataset('data/processed', 'data/processed/train_pairs.json')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()

    print(f"[2] Training on {device}")
    for epoch in range(40):
        model.train()
        loss_sum = correct = total = 0
        for ref, sample, label in tqdm(loader, desc=f"Epoch {epoch+1}"):
            ref, sample, label = ref.to(device), sample.to(device), label.to(device)
            opt.zero_grad()
            pred = model(ref, sample)
            loss = crit(pred, label)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            correct += ((pred > 0.5) == label).sum().item()
            total += label.size(0)
        print(f"   Loss: {loss_sum/len(loader):.4f} | Acc: {correct/total:.4f}")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/siamese_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), "models/siamese_final.pth")
    print("[2] DONE!")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    train()