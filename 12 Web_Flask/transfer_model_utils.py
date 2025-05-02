import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# 1. CSV 로드
df = pd.read_csv('info.csv').drop(columns=['Unnamed: 0']).fillna('')

# 2. 모델 로드
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 3. 텍스트 결합
df['full_text'] = (
    df['name'] + " " +
    df['category'] + " " +
    df['company']  + " " +
    df['first_prize'] + " " +
    df['qual']  

    
)

# 4. 임베딩
print("임베딩 생성 중...")
corpus_embeddings = model.encode(df['full_text'].tolist(), convert_to_tensor=True)

# 5. 사용자 입력
user_input = input("추천받을 카테고리를 입력하세요: ").strip()

# 6. 쿼리 임베딩
query_embedding = model.encode(user_input, convert_to_tensor=True)

# 7. 유사도 계산
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# 8. 상위 5개
topk = torch.topk(cos_scores, k=5)


top_scores = topk.values.tolist()      # [float, float, ...]
top_indices = topk.indices.tolist()    # [int, int, ...]

print(f"\n▶ 카테고리 '{user_input}' 기반 추천 공모전 TOP 5:")
for score, idx in zip(top_scores, top_indices):
    # idx는 이제 int
    row = df.iloc[idx]
    print(f"  카테고리: {row['category']}")
    print(f"  제목: {row['name']}")
    print(f"  주최/주관: {row['company']}")
    print(f"  참가대상: {row['qual']}")
    print(f"  접수기간: {row['register_start']} ~ {row['register_end']}")
    print(f"  총상금: {row['first_prize']}")
    print(f"  유사도: {score:.4f}\n")


# 1) 임베딩 계산 후 저장
# torch.save(corpus_embeddings, 'corpus_embeddings.pt')
# print("임베딩 텐서를 'corpus_embeddings.pt'로 저장했습니다.")


## 임베딩 로딩 후 사용 예시

## Load model and compute embeddings

# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# df = pd.read_csv('info.csv').drop(columns=['Unnamed: 0']).fillna('')

## --- 2. 이후 추천 스크립트: 로드한 임베딩 재사용 ---
## Re-load embeddings (no need to re-encode the entire corpus)
# loaded_embeddings = torch.load('corpus_embeddings.pt')
# print(f"✔ 불러온 임베딩 shape: {loaded_embeddings.shape}")

## Get user input and recommend
# user_input = input("추천받을 카테고리를 입력하세요: ").strip()
# query_embedding = model.encode(user_input, convert_to_tensor=True)

## Compute cosine similarity with loaded embeddings
# cos_scores = util.cos_sim(query_embedding, loaded_embeddings)[0]
# topk = torch.topk(cos_scores, k=5)
# top_scores = topk.values.tolist()
# top_indices = topk.indices.tolist()

# print(f"\n▶ '{user_input}' 추천 공모전 TOP 5:")
# for score, idx in zip(top_scores, top_indices):
#     row = df.iloc[idx]
#     print(f"- 제목: {row['name']} | 주최: {row['company']} | 기간: {row['register_start']}~{row['register_end']} | 유사도: {score:.4f}")