# Example: Saving and loading `corpus_embeddings` for later use

import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd






model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

df = pd.read_csv('info.csv').drop(columns=['Unnamed: 0']).fillna('')


loaded_embeddings = torch.load('corpus_embeddings.pt')
print(f"✔ 불러온 임베딩 shape: {loaded_embeddings.shape}")


user_input = input("추천받을 카테고리를 입력하세요: ").strip()
query_embedding = model.encode(user_input, convert_to_tensor=True)


cos_scores = util.cos_sim(query_embedding, loaded_embeddings)[0]
topk = torch.topk(cos_scores, k=5)
top_scores = topk.values.tolist()
top_indices = topk.indices.tolist()

print(f"\n▶ '{user_input}' 추천 공모전 TOP 5:")
for score, idx in zip(top_scores, top_indices):
    row = df.iloc[idx]
    print(f"- 제목: {row['name']} | 주최: {row['company']} | 기간: {row['register_start']}~{row['register_end']} | 유사도: {score:.4f}")
