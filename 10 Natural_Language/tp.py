import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ================================
# 0. 장치 설정 (CUDA)
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================================
# 1. 엑셀 파일에서 원문/번역문 읽어오기
# ================================
xlsx_files = [    
    r"F:\KDT7\11_torchtext\team_project\data\1_구어체(1).xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\1_구어체(2).xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\2_대화체.xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\3_문어체_뉴스(1)_200226.xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\3_문어체_뉴스(2).xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\3_문어체_뉴스(3).xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\3_문어체_뉴스(4).xlsx",
    r"F:\KDT7\11_torchtext\team_project\data\4_문어체_한국문화.xlsx"
    ]

ko_list = []
en_list = []

for file_path in tqdm(xlsx_files, desc="Reading XLSX files"):
    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        continue
    df = pd.read_excel(file_path)
    
    if "원문" not in df.columns or "번역문" not in df.columns:
        print(f"[WARN] Missing '원문'/'번역문' columns in {file_path}")
        continue

    for i, row in df.iterrows():
        src_text = str(row["원문"]).strip()
        tgt_text = str(row["번역문"]).strip()
        if src_text and tgt_text:
            ko_list.append(src_text)
            en_list.append(tgt_text)

print("Total loaded pairs:", len(ko_list))

# ================================
# 2. Vocab(단어 사전) 구성
# ================================
class Vocab:
    """
    <pad> =0, <sos>=1, <eos>=2, <unk>=3
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.word2idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {0: self.pad_token, 1: self.sos_token,
                         2: self.eos_token, 3: self.unk_token}
        self.n_words = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def tokenize(self, text):
        return text.split()  

    def build_vocab(self, text_list, max_size=None, min_freq=1):
        freq = collections.Counter()
        for line in text_list:
            tokens = self.tokenize(line)
            freq.update(tokens)
        sorted_items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for w, f in sorted_items:
            if f < min_freq:
                break
            if max_size is not None and self.n_words >= max_size:
                break
            self.add_word(w)

    def encode(self, text, add_sos=True, add_eos=True):
        tokens = self.tokenize(text)
        seq = []
        if add_sos:
            seq.append(self.word2idx[self.sos_token])
        for t in tokens:
            if t in self.word2idx:
                seq.append(self.word2idx[t])
            else:
                seq.append(self.word2idx[self.unk_token])
        if add_eos:
            seq.append(self.word2idx[self.eos_token])
        return seq

    def decode(self, seq):
        tokens = []
        for idx in seq:
            if idx == self.word2idx[self.sos_token]:
                continue
            if idx == self.word2idx[self.eos_token]:
                break
            if idx == self.word2idx[self.pad_token]:
                continue
            tokens.append(self.idx2word.get(idx, self.unk_token))
        return " ".join(tokens)


ko_vocab = Vocab()
en_vocab = Vocab()


ko_vocab.build_vocab(ko_list, max_size=5000, min_freq=5)
en_vocab.build_vocab(en_list, max_size=5000, min_freq=5)

print("Korean Vocab size:", ko_vocab.n_words)
print("English Vocab size:", en_vocab.n_words)
with open("src_vocab.pkl", "wb") as f:
    pickle.dump(ko_vocab, f)


with open("tgt_vocab.pkl", "wb") as f:
    pickle.dump(en_vocab, f)
# ================================
# 3. Dataset / DataLoader
# ================================
class TranslationDataset(Dataset):
    def __init__(self, src_list, tgt_list, src_vocab, tgt_vocab):
        assert len(src_list) == len(tgt_list)
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        src_text = self.src_list[idx]
        tgt_text = self.tgt_list[idx]
        src_ids = self.src_vocab.encode(src_text, True, True)
        tgt_ids = self.tgt_vocab.encode(tgt_text, True, True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    PAD_IDX = 0
    batch_size = len(batch)

    src_padded = torch.full((batch_size, max_src_len), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt_len), PAD_IDX, dtype=torch.long)

    for i in range(batch_size):
        src_padded[i, :src_lens[i]] = src_batch[i]
        tgt_padded[i, :tgt_lens[i]] = tgt_batch[i]

    return {"src": src_padded, "tgt": tgt_padded}

# 훈련/검증 분리
total_size = len(ko_list)
train_size = int(total_size * 0.8)
indices = np.arange(total_size)
np.random.shuffle(indices)

train_idx = indices[:train_size]
valid_idx = indices[train_size:]

train_src = [ko_list[i] for i in train_idx]
train_tgt = [en_list[i] for i in train_idx]
valid_src = [ko_list[i] for i in valid_idx]
valid_tgt = [en_list[i] for i in valid_idx]

train_dataset = TranslationDataset(train_src, train_tgt, ko_vocab, en_vocab)
valid_dataset = TranslationDataset(valid_src, valid_tgt, ko_vocab, en_vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

print("Train dataset:", len(train_dataset), "Valid dataset:", len(valid_dataset))

# ================================
# 4. RNN(LSTM) (5개 레이어)
#    Encoder(2층) + Decoder(3층)
#    브리지(bridging) 로직 추가
# ================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, pad_idx=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=emb_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, src):
        embedded = self.embedding(src)  # (B,src_len,emb_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        # hidden, cell: (n_layers, B, hidden_dim)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, pad_idx=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_embeddings=output_dim,
                                      embedding_dim=emb_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_step, hidden, cell):
        # input_step: (B,)
        # hidden, cell: (n_layers, B, hidden_dim)
        input_step = input_step.unsqueeze(1)  # (B,1)
        embedded = self.embedding(input_step) # (B,1,emb_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # outputs: (B,1,hidden_dim)
        prediction = self.fc_out(outputs.squeeze(1))  # (B, output_dim)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0, sos_idx=1, eos_idx=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src, tgt):
        """
        src: (B, src_len)
        tgt: (B, tgt_len)
        """
        batch_size, tgt_len = tgt.size()
        output_dim = self.decoder.fc_out.out_features

        # 1) 인코더
        enc_hidden, enc_cell = self.encoder(src)
        # enc_hidden, enc_cell => shape (enc_layers, B, hidden_dim)

        # 2) 브리지(bridging): 디코더 레이어 수와 맞춤
        enc_layers = enc_hidden.size(0)
        dec_layers = self.decoder.n_layers
        if dec_layers > enc_layers:
            # 예: 인코더2층 -> 디코더3층 -> 복제
            diff = dec_layers - enc_layers
            # 마지막 레이어(인덱스 -1)를 복제
            enc_hidden_extra = enc_hidden[-1:].repeat(diff, 1, 1)  # (diff,B,H)
            enc_cell_extra = enc_cell[-1:].repeat(diff, 1, 1)
            hidden = torch.cat([enc_hidden, enc_hidden_extra], dim=0)
            cell = torch.cat([enc_cell, enc_cell_extra], dim=0)
        elif dec_layers < enc_layers:
            # 인코더 레이어가 더 많다면 잘라냄
            hidden = enc_hidden[:dec_layers, :, :]
            cell = enc_cell[:dec_layers, :, :]
        else:
            hidden = enc_hidden
            cell = enc_cell

        # 3) 디코더
        outputs = torch.zeros(batch_size, tgt_len, output_dim, device=src.device)
        input_step = tgt[:, 0]  # <sos>
        for t in range(1, tgt_len):
            prediction, hidden, cell = self.decoder(input_step, hidden, cell)
            outputs[:, t, :] = prediction
            # teacher forcing
            input_step = tgt[:, t]
        return outputs

# 2층 인코더 + 3층 디코더 = 총 5층
ENC_LAYERS = 2
DEC_LAYERS = 3
HIDDEN_DIM = 512
EMB_DIM = 256

encoder = Encoder(input_dim=ko_vocab.n_words,
                  emb_dim=EMB_DIM,
                  hidden_dim=HIDDEN_DIM,
                  n_layers=ENC_LAYERS,
                  pad_idx=0)
decoder = Decoder(output_dim=en_vocab.n_words,
                  emb_dim=EMB_DIM,
                  hidden_dim=HIDDEN_DIM,
                  n_layers=DEC_LAYERS,
                  pad_idx=0)

model = Seq2Seq(encoder, decoder, pad_idx=0, sos_idx=1, eos_idx=2).to(DEVICE)

# ================================
# 5. 학습 준비 (Optimizer, Loss)
# ================================
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad=0 무시

def accuracy_fn(pred, y):
    """
    pred: (B, seq_len, vocab_size)
    y   : (B, seq_len)
    """
    pred_idx = pred.argmax(dim=-1)  
    mask = (y != 0).float()         # pad=0 제외
    correct = ((pred_idx == y).float() * mask).sum()
    total = mask.sum()
    return (correct/(total+1e-8)).item()

# ================================
# 6. 학습/검증 루프
# ================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch in tqdm(loader, desc="Training"):
        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(src, tgt) 

        # t=1.. 사용 (t=0은 <sos>)
        pred_reshaped = outputs[:, 1:, :].reshape(-1, en_vocab.n_words)
        tgt_reshaped = tgt[:, 1:].reshape(-1)

        loss = criterion(pred_reshaped, tgt_reshaped)
        acc = accuracy_fn(outputs[:, 1:, :], tgt[:, 1:])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        count += 1
    return total_loss / count, total_acc / count

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt"].to(DEVICE)
            outputs = model(src, tgt)

            pred_reshaped = outputs[:, 1:, :].reshape(-1, en_vocab.n_words)
            tgt_reshaped = tgt[:, 1:].reshape(-1)

            loss = criterion(pred_reshaped, tgt_reshaped)
            acc = accuracy_fn(outputs[:, 1:, :], tgt[:, 1:])
            total_loss += loss.item()
            total_acc += acc
            count += 1
    return total_loss/count, total_acc/count

# ================================
# 7. 실제 학습
# ================================
# EPOCHS = 15
# best_val_acc = -1.0

# train_loss_list = []
# train_acc_list  = []
# val_loss_list   = []
# val_acc_list    = []

# for epoch in range(1, EPOCHS+1):
#     print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
#     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
#     val_loss, val_acc = eval_one_epoch(model, valid_loader, criterion)

#     train_loss_list.append(train_loss)
#     train_acc_list.append(train_acc)
#     val_loss_list.append(val_loss)
#     val_acc_list.append(val_acc)

#     print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
#           f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

#     # val_acc가 좋아지면 모델 저장
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "best_seq2seq_model.pth")
#         print("  >> Best model saved (val_acc improved).")

# ================================
# 8. 학습 결과 시각화
# ================================
# plt.figure(figsize=(10,4))

# plt.subplot(1,2,1)
# plt.plot(train_loss_list, label='Train Loss')
# plt.plot(val_loss_list,   label='Valid Loss')
# plt.title('Loss')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(train_acc_list, label='Train Acc')
# plt.plot(val_acc_list,   label='Valid Acc')
# plt.title('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.savefig("training_plot.png")
# plt.show()

# ================================
# 9. 번역 함수 (모델 로드 후 사용)
# ================================
def translate_korean_sentence(model, sentence, max_len=50):
    model.eval()
    with torch.no_grad():
        # 문장 -> 인덱스
        src_ids = ko_vocab.encode(sentence, True, True)
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, seq_len)

        # 인코더
        enc_hidden, enc_cell = model.encoder(src_tensor)

        # 브리지 적용
        enc_layers = enc_hidden.size(0)
        dec_layers = model.decoder.n_layers
        if dec_layers > enc_layers:
            diff = dec_layers - enc_layers
            enc_hidden_extra = enc_hidden[-1:].repeat(diff, 1, 1)
            enc_cell_extra = enc_cell[-1:].repeat(diff, 1, 1)
            hidden = torch.cat([enc_hidden, enc_hidden_extra], dim=0)
            cell = torch.cat([enc_cell, enc_cell_extra], dim=0)
        elif dec_layers < enc_layers:
            hidden = enc_hidden[:dec_layers,:,:]
            cell   = enc_cell[:dec_layers,:,:]
        else:
            hidden, cell = enc_hidden, enc_cell

        # 디코더
        outputs = []
        input_step = torch.tensor([en_vocab.word2idx["<sos>"]], dtype=torch.long).to(DEVICE)  # (1,)
        for t in range(max_len):
            pred, hidden, cell = model.decoder(input_step, hidden, cell)
            next_token = pred.argmax(dim=-1).item()
            if next_token == en_vocab.word2idx["<eos>"]:
                break
            if next_token == en_vocab.word2idx["<pad>"]:
                break
            outputs.append(next_token)
            input_step = torch.tensor([next_token], dtype=torch.long).to(DEVICE)

        return en_vocab.decode(outputs)

def translate_file(model, txt_path):
    if not os.path.exists(txt_path):
        print("File not found:", txt_path)
        return
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            print("[KO]", line_strip)
            print("[EN]", "")
            continue
        eng_line = translate_korean_sentence(model, line_strip)
        print("[KO]", line_strip)
        print("[EN]", eng_line, "\n")

## 모델 불러와 번역
# model.load_state_dict(torch.load("best_seq2seq_model.pth"))
# user_input = input("한국어 입력: ")
# print("번역 결과:", translate_korean_sentence(model, user_input))
