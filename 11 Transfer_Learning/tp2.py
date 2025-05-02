from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.ops import box_iou
from PIL import Image
from lxml import etree
import matplotlib.pyplot as plt
from tqdm import tqdm

from effdet import create_model, DetBenchTrain, DetBenchPredict

# -----------------------------------------------------------------------------
# Utility – AMP‑autocast
# -----------------------------------------------------------------------------
if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
    def autocast_ctx(enabled: bool, device_type: str):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
else:
    from torch.amp import autocast as _old_autocast
    def autocast_ctx(enabled: bool, device_type: str):
        return _old_autocast(enabled=enabled)

from torch.amp import GradScaler

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class XRAYDataset(torch.utils.data.Dataset):
    CLASSES = ['Firecracker', 'Hammer', 'NailClippers', 'Spanner', 'Thinner', 'ZippoOil']
    CLASS2IDX = {n: i+1 for i, n in enumerate(CLASSES)}; CLASS2IDX['unknown'] = 0

    def __init__(self, img_root: str | Path, xml_root: str | Path, transform=None):
        self.img_root, self.xml_root, self.transform = Path(img_root), Path(xml_root), transform
        self.items = [(p, self.xml_root / f"{p.stem}.xml")
                      for p in self.img_root.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, xml_path = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        boxes, labels = [], []
        root = etree.parse(str(xml_path)).getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text; name = name if name in self.CLASSES else 'unknown'
            bb = obj.find('bndbox')
            xmin, ymin = int(bb.find('xmin').text)//3, int(bb.find('ymin').text)//3
            xmax, ymax = int(bb.find('xmax').text)//3, int(bb.find('ymax').text)//3
            boxes.append([xmin, ymin, xmax, ymax]); labels.append(self.CLASS2IDX[name])
            
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        if self.transform: img = self.transform(img)
        return img, target

# -----------------------------------------------------------------------------
# Pre‑processing
# -----------------------------------------------------------------------------
TRANSFORM = T.Compose([
    T.Resize((360, 640)),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
])

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    batch_tgts = []
    for t in tgts:
        t_new = {'bbox': t['boxes'], 'cls': t['labels']}
        t_new['label_num_positives'] = torch.tensor([len(t['labels'])], dtype=torch.float32)
        batch_tgts.append(t_new)
    return torch.stack(imgs), batch_tgts

# -----------------------------------------------------------------------------
# Metric
# -----------------------------------------------------------------------------
def val_accuracy(pred, targ, thr=0.5):
    if targ['bbox'].numel() == 0:
        return 1.0
    ious = box_iou(pred['bbox'], targ['bbox'])
    best_iou, idx = ious.max(dim=0)
    correct = (best_iou >= thr) & (pred['cls'][idx] == targ['cls'])
    return correct.float().mean().item()

# -----------------------------------------------------------------------------
# Train / Validate loops
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train(); epoch_loss = 0.0
    for imgs, tgts in tqdm(loader, desc='Train', ncols=80):
        imgs = imgs.to(device)
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        # 리스트 형태의 tgts를 모델이 기대하는 딕셔너리 형태로 변환
        batch_tgt = {}
        for key in tgts[0].keys():
            batch_tgt[key] = torch.stack([t[key] for t in tgts])

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(scaler is not None, device_type=device.type):
            outputs = model(imgs, batch_tgt)
            loss = outputs['loss']
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    return epoch_loss / len(loader.dataset)


def validate(model_eval, loader, device):
    model_eval.eval(); acc_sum = 0.0
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc='Val', ncols=80):
            imgs = imgs.to(device)
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            outputs = model_eval(imgs)
            for out, tgt in zip(outputs, tgts):
                pred = {'bbox': out['boxes'], 'cls': out['labels']}
                acc_sum += val_accuracy(pred, tgt)
    return acc_sum / len(loader)

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------
def plot_history(hist, out_path):
    ep = range(1, len(hist['train_loss'])+1)
    plt.figure(); plt.plot(ep, hist['train_loss'], label='TrainLoss'); plt.plot(ep, hist['val_acc'], label='ValAcc')
    plt.xlabel('Epoch'); plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    train_img = r'F:\KDT7\12_trans\team_project\xray-img\Astrophysics\Single_Default' 
    train_xml = r'F:\KDT7\12_trans\team_project\xray-img\Annotation\Train\Pascal\Astrophysics_SingleDefaultOnly2'
    val_img = r'F:\KDT7\12_trans\team_project\xray-img\Astrophysics_val\Single_Default' 
    val_xml = r'F:\KDT7\12_trans\team_project\xray-img\Annotation\eval\Pascal\Astrophysics_SingleDefaultOnly'

    EPOCHS, TRAIN_BS, VAL_BS, LR = 50, 16, 2, 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = DataLoader(XRAYDataset(train_img, train_xml, TRANSFORM), batch_size=TRAIN_BS, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(XRAYDataset(val_img, val_xml, TRANSFORM), batch_size=VAL_BS, shuffle=False, collate_fn=collate_fn)

    # DetBenchTrain 제거
    base_model = create_model('tf_efficientdet_d1', bench_task=None, num_classes=7, pretrained=True, image_size=(360, 640))

# 학습용
    model = DetBenchTrain(base_model).to(device)

# 평가용
    eval_model = DetBenchPredict(base_model).to(device)
    run_dir = Path('runs')/f"xray_d4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"; run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir.as_posix())
    history, best = {'train_loss': [], 'val_acc': []}, 0.0

    # 평가용 모델 생성 및 파라미터 로드


    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scaler = GradScaler() if device.type == 'cuda' else None        
        tr_loss = train_one_epoch(model, train_dl, optimizer, scaler, device)

        eval_model.load_state_dict(model.state_dict())  # 파라미터 동기화
        eval_model.eval()
        val_acc = validate(eval_model, val_dl, device)

        history['train_loss'].append(tr_loss); history['val_acc'].append(val_acc)
        writer.add_scalar('Loss/train', tr_loss, epoch); writer.add_scalar('Acc/val', val_acc, epoch)
        plot_history(history, run_dir/'training_plot.png')

        if val_acc > best:
            best = val_acc; torch.save(model.state_dict(), run_dir/'best_model.pth')
            print(f"New best Val-Acc: {best:.4f}")
    writer.close(); print('Training finished.')

if __name__ == '__main__':
    main()
