import os
import cv2
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torchinfo import summary


############################################
# 1) Dataset
############################################
class FungiDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        img_path = os.path.join(self.img_dir, filename)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"[오류] 이미지 로드 실패: {img_path}")

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 128 x 128 리사이즈
        img_rgb = cv2.resize(img_rgb, (128, 128))

        label = int(row['region_idx'])

        if self.transform:
            img_rgb = self.transform(img_rgb)

        return img_rgb, label


############################################
# 2) CNN 모델
############################################
class FungiCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   64,   kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,  128,  kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)
        self.dropout_rate = 0.25

    def forward(self, x):
        # conv1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # conv2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # conv3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # conv4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # conv5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 전역풀링 & FC
        x = self.global_pool(x)  # shape=(N,1024,1,1)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


############################################
# 3) 평가 함수
############################################
def evaluate_model(model, loader, device):

    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            loss = criterion(outs, labels)
            total_loss += loss.item() * imgs.size(0)

            _, preds = torch.max(outs,1)
            correct += (preds==labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / (total if total else 1)
    acc      = correct / (total if total else 1)
    return avg_loss, acc


############################################
# 4) “train 기준으로 best model 저장” & “기존 모델 이어 학습” & “테스트 acc≥0.9시 중단”
############################################
def train_fungi_cnn_with_scheduler_resume(
    train_csv, valid_csv, test_csv,
    train_img_dir, valid_img_dir, test_img_dir,
    epochs=50, patience=10, batch_size=32, lr=1e-3,
    model_save_path=r"F:\KDT7\fungi_cnn_best_trainBased.pth"
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # (A) CSV 로드 + 라벨인코딩
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    df_test  = pd.read_csv(test_csv)

    # 결측치 -> '알수없음'
    for df_ in [df_train, df_valid, df_test]:
        df_['biogeographicalRegion'] = df_['biogeographicalRegion'].fillna('알수없음')

    # 라벨인코딩
    le = LabelEncoder()
    all_regions = pd.concat([df_train['biogeographicalRegion'],
                             df_valid['biogeographicalRegion'],
                             df_test['biogeographicalRegion']], ignore_index=True)
    le.fit(all_regions)

    df_train['region_idx'] = le.transform(df_train['biogeographicalRegion'])
    df_valid['region_idx'] = le.transform(df_valid['biogeographicalRegion'])
    df_test ['region_idx'] = le.transform(df_test ['biogeographicalRegion'])

    # 라벨인코더 저장 (옵션)
    with open("fungi_label_encoder.pkl","wb") as f:
        pickle.dump(le, f)

    # (B) Dataset/DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    # 검증/테스트는 통상 강한 증강 없이 기본만
    transform_valtest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_ds = FungiDataset(df_train, train_img_dir, transform)
    valid_ds = FungiDataset(df_valid, valid_img_dir, transform_valtest)
    test_ds  = FungiDataset(df_test,  test_img_dir,  transform_valtest)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # (C) 모델 준비
    num_classes = len(le.classes_)
    model = FungiCNN(num_classes).to(device)
    

    # ✅ 모델 summary 출력
    print("\n[모델 구조 요약]")
    summary(model)

    # 만약 기존 모델이 존재하면 불러오기(이어 학습)
    if os.path.exists(model_save_path):
        print(f"기존 모델 발견: {model_save_path} 를 로드합니다...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("기존 모델이 없으므로 새로 학습을 시작합니다.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train_acc 기준 scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 기록
    train_loss_list, train_acc_list = [], []
    val_loss_list,   val_acc_list   = [], []

    best_val_loss = 0.4305
    best_epoch     = 0
    stop_flag      = False

    # (D) 학습 루프
    for epoch in range(1, epochs+1):
        # 1) train
        model.train()
        running_loss=0.0
        correct=0
        total=0

        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outs,1)
            correct  += (preds==labels).sum().item()
            total    += labels.size(0)

        train_loss = running_loss / len(train_ds)
        train_acc  = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 2) val
        val_loss, val_acc = evaluate_model(model, valid_loader, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # 3) scheduler step (train_acc 모니터링)
        scheduler.step(val_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        print(f"Epoch[{epoch}/{epochs}] trainLoss={train_loss:.4f}, trainAcc={train_acc:.4f} | "
              f"valLoss={val_loss:.4f}, valAcc={val_acc:.4f} | LR={current_lr:.6f}")

        # (저장 기준) "train_acc"가 이전보다 개선되었다면 => 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch     = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"[BEST] train_acc={val_loss:.4f} (epoch={epoch}) => 모델 저장")

        # (조기중단) patience=10동안 train_acc 개선 없으면 중단
        if (epoch - best_epoch) >= patience:
            print(f"** {patience} epochs no improvement => Early stopping. **")
            stop_flag = True

        # (추가) test정확도 0.9 넘으면 즉시 중단
        #   - 매 epoch가 끝날 때마다 test를 돌리면 시간이 늘어나지만 예시를 위해
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        if test_acc >= 0.98:
            print(f"** Test Accuracy={test_acc:.4f} >= 0.98 => 조기 종료합니다. **")
            stop_flag = True

        if stop_flag:
            break

    # (E) 그래프 그리기
    ep_range = range(1, epoch+1)

    fig, axs = plt.subplots(1,2, figsize=(12,5))
    # Accuracy
    axs[0].plot(ep_range, train_acc_list, 'b-', label='Train Acc')
    axs[0].plot(ep_range, val_acc_list,   'r-', label='Val Acc')
    axs[0].set_title("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Acc")
    axs[0].legend()

    # Loss
    axs[1].plot(ep_range, train_loss_list, 'b-', label='Train Loss')
    axs[1].plot(ep_range, val_loss_list,   'r-', label='Val   Loss')
    axs[1].set_title("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # (F) 학습 마지막에, 저장된 best 모델 로드하여 final test
    print("\n[Load best model & Evaluate TEST one last time]")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"=== Final Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f} ===")


##########################
# 실행 예시
##########################
if __name__=="__main__":
    train_fungi_cnn_with_scheduler_resume(
        train_csv=r"F:\KDT7\10_deep_learning\team_project\fungi-clef-2025\metadata\FungiTastic-FewShot\FungiTastic-FewShot-Train.csv",
        valid_csv=r"F:\KDT7\10_deep_learning\team_project\fungi-clef-2025\metadata\FungiTastic-FewShot\FungiTastic-FewShot-Val.csv",
        test_csv =r"F:\KDT7\10_deep_learning\team_project\fungi-clef-2025\metadata\FungiTastic-FewShot\FungiTastic-FewShot-Test.csv",

        train_img_dir=r"F:\KDT7\10_deep_learning\team_project\fungi-clef-2025\images\FungiTastic-FewShot\train\300p",
        valid_img_dir=r"F:\KDT7\10_deep_learning\team_project\fungi-clef-2025\images\FungiTastic-FewShot\val\300p",
        test_img_dir =r"F:\KDT7\10_deep_learning\team_project\fungi-clef-2025\images\FungiTastic-FewShot\test\300p",

        epochs=1000,            # 최대 epoch
        patience=10,           # 10번동안 개선 없으면 중단
        batch_size=16,
        lr=1e-3,
        model_save_path=r"F:\KDT7\fungi_cnn_best_trainBased.pth"  # 여기에서 기존 모델 이어학습 + 저장
    )
