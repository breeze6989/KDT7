#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io

# 한글 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
import pickle
import os

########################################
# 0) Vocab 클래스 (pickle 호환)
########################################
class Vocab:
    """
    tp_final_complete_all_fixed.py 내와 동일한 구조/이름
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.word2idx = {self.pad_token:0, self.sos_token:1, self.eos_token:2, self.unk_token:3}
        self.idx2word = {0:self.pad_token, 1:self.sos_token, 2:self.eos_token, 3:self.unk_token}
        self.n_words = 4

########################################
# 1) Encoder, Decoder (2층 vs 3층) + bridging
########################################
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=2, pad_idx=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True)

    def forward(self, src):
        # src: (B, src_len)
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell  # hidden/cell: (n_layers,B,hidden_dim)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=3, pad_idx=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_step, hidden, cell):
        # input_step: (B,)
        input_step = input_step.unsqueeze(1)  # (B,1)
        embedded = self.embedding(input_step) # (B,1,emb_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
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

    def bridging(self, enc_hidden, enc_cell):
        """
        브리지 로직:
        - 인코더: n_layers=2
        - 디코더: n_layers=3
        => 디코더가 더 많으므로 마지막 레이어( enc_hidden[-1:] ) 복제
        """
        enc_layers = enc_hidden.size(0)
        dec_layers = self.decoder.n_layers
        if dec_layers > enc_layers:
            diff = dec_layers - enc_layers
            enc_hidden_extra = enc_hidden[-1:].repeat(diff, 1, 1)  # (diff,B,H)
            enc_cell_extra   = enc_cell[-1:].repeat(diff, 1, 1)
            hidden = torch.cat([enc_hidden, enc_hidden_extra], dim=0)
            cell   = torch.cat([enc_cell,   enc_cell_extra],   dim=0)
        elif dec_layers < enc_layers:
            hidden = enc_hidden[:dec_layers, :, :]
            cell   = enc_cell[:dec_layers, :, :]
        else:
            hidden, cell = enc_hidden, enc_cell
        return hidden, cell

    def translate(self, src, max_len=50):
        """
        단일 배치(1) 토큰 단순 번역 (Greedy)
        src: shape (1, src_len)
        """
        # 1) 인코더
        enc_hidden, enc_cell = self.encoder(src)
        # 2) bridging
        hidden, cell = self.bridging(enc_hidden, enc_cell)

        # 3) 디코더 loop
        input_token = torch.tensor([self.sos_idx], dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            out, hidden, cell = self.decoder(input_token, hidden, cell)
            top1 = out.argmax(dim=1)  # (1,)
            if top1.item() == self.eos_idx:
                break
            outputs.append(top1.item())
            input_token = top1
        return outputs

########################################
# 2) Flask 앱
########################################
app = Flask(__name__)

# HTML 템플릿
HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>한국어 -> 영어 번역</title>
</head>
<body>
  <h2>한국어 문장 입력 </h2>
  <form method="POST">
    <input type="text" name="user_input" style="width:300px;">
    <input type="submit" value="번역">
  </form>
  {% if translation %}
    <h3>번역 결과:</h3>
    <p>{{ translation }}</p>
  {% endif %}
</body>
</html>
"""

########################################
# 3) pth, pkl 파일 로드
########################################
MODEL_PATH = "best_seq2seq_model.pth"
SRC_VOCAB_PATH = "src_vocab.pkl"
TGT_VOCAB_PATH = "tgt_vocab.pkl"

with open(SRC_VOCAB_PATH, "rb") as f:
    src_vocab = pickle.load(f)
with open(TGT_VOCAB_PATH, "rb") as f:
    tgt_vocab = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = len(src_vocab.word2idx)
OUTPUT_DIM = len(tgt_vocab.word2idx)
EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 2  # 인코더 2층
DEC_LAYERS = 3  # 디코더 3층

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, n_layers=ENC_LAYERS)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, n_layers=DEC_LAYERS)
model = Seq2Seq(encoder, decoder, pad_idx=0, sos_idx=1, eos_idx=2).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
model.eval()

########################################
# 4) 라우팅
########################################
@app.route("/", methods=["GET","POST"])
def translate():
    translation = ""
    if request.method == "POST":
        text = request.form.get("user_input","").strip()
        if text:
            tokens = text.split()
            # src_vocab["<sos>"], ...
            src_ids = [src_vocab.word2idx["<sos>"]] + [
                src_vocab.word2idx.get(t, src_vocab.word2idx["<unk>"])
                for t in tokens
            ] + [src_vocab.word2idx["<eos>"]]

            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                output_indices = model.translate(src_tensor, max_len=50)

            # 인덱스 -> 단어
            idx2word = {v:k for k,v in tgt_vocab.word2idx.items()}
            words = [idx2word.get(i, "<unk>") for i in output_indices]
            translation = " ".join(words)

    return render_template_string(HTML, translation=translation)


if __name__ == "__main__":
    app.run(port=8080, debug=False)
