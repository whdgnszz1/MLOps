import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# MovieLens 데이터셋 로드
data = load_dataset("nbtpj/movielens-1m-ratings")["train"].shuffle(seed=10)
n_chunks = 100
chunk_size = len(data) // n_chunks
if len(data) % n_chunks != 0: chunk_size += 1

movielens_df = pd.DataFrame()

for start in tqdm(range(0, len(data), chunk_size)):
    end = start + chunk_size
    chunk_df = pd.DataFrame(data[start:end])
    movielens_df = pd.concat([movielens_df, chunk_df], ignore_index=True)

# user_id를 문자열로 변환 및 필요한 열만 선택
movielens_df["user_id"] = movielens_df["user_id"].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
movielens_df = movielens_df[["user_id", "movie_id", "user_rating"]]

# 결과 확인 (상위 5개 행 출력)
print("MovieLens 데이터셋 예시:")
print(movielens_df.head())