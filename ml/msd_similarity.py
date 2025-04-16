import pandas as pd
from datasets import load_dataset
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# MovieLens 데이터셋 로드
data = load_dataset("nbtpj/movielens-1m-ratings")["train"].shuffle(seed=10).select(range(500000))
movielens_df = pd.DataFrame(data)[["user_id", "movie_id", "user_rating"]]

# 사용자와 영화 필터링 (최소 평점 수 기준)
min_user_ratings = 20
min_movie_ratings = 10
user_counts = movielens_df['user_id'].value_counts()
movie_counts = movielens_df['movie_id'].value_counts()
filtered_df = movielens_df[
    movielens_df['user_id'].isin(user_counts[user_counts >= min_user_ratings].index) &
    movielens_df['movie_id'].isin(movie_counts[movie_counts >= min_movie_ratings].index)
    ].copy()

# Surprise 데이터셋 준비
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(filtered_df[['user_id', 'movie_id', 'user_rating']], reader)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=10)

# MSD 기반 KNN 모델 설정
sim_options = {
    "name": "msd",
    "user_based": True,
    "min_support": 5
}
algo = KNNBasic(k=20, sim_options=sim_options)

# 모델 학습
algo.fit(train_data)

# 예측 및 평가
predictions = algo.test(test_data)
rmse = accuracy.rmse(predictions)
print(f"MSD 기반 KNN RMSE: {rmse}")

# 예측 vs 실제 평점 시각화
actual_ratings = [pred.r_ui for pred in predictions]
predicted_ratings = [round(pred.est) for pred in predictions]

plt.figure(figsize=(10, 5))
plt.hist(predicted_ratings, bins=5, alpha=0.5, label="예측", color="#fc1c49")
plt.hist(actual_ratings, bins=5, alpha=0.5, label="실제", color="#00a67d")
plt.title("예측 vs 실제 평점 분포 (MSD 기반 KNN)")
plt.xlabel("평점")
plt.ylabel("빈도")
plt.legend()
plt.show()
