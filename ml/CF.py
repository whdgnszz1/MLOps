import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

# 데이터 로드 및 전처리
data = load_dataset("nbtpj/movielens-1m-ratings")["train"].shuffle(seed=10)
n_chunks = 100
chunk_size = len(data) // n_chunks
if len(data) % n_chunks != 0: chunk_size += 1

movielens_df = pd.DataFrame()

for start in tqdm(range(0, len(data), chunk_size)):
    end = start + chunk_size
    chunk_df = pd.DataFrame(data[start:end])
    movielens_df = pd.concat([movielens_df, chunk_df], ignore_index=True)

# user_id를 문자열로 변환
movielens_df["user_id"] = movielens_df["user_id"].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
movielens_df = movielens_df[["user_id", "movie_id", "user_rating"]]

# 데이터 분할
train_data, test_data = train_test_split(movielens_df, test_size=0.02, random_state=10)

# 훈련 데이터로 사용자-아이템 매트릭스 생성
user_item_matrix = train_data.pivot_table(index="user_id", columns="movie_id", values="user_rating")

# 사용자 유사도 계산
user_similarity = cosine_similarity(csr_matrix(user_item_matrix.fillna(0)))
user_similarity = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# 사용자별 평균 평점 계산
user_means = user_item_matrix.mean(axis=1)


# KNN 기반 평점 예측 함수
def predict_rating_knn(user_id: str, movie_id: str, k=20):
    if movie_id not in user_item_matrix.columns:
        return train_data["user_rating"].mean()
    movie_ratings = user_item_matrix[movie_id]
    user_similarities = user_similarity[user_id]
    valid_indices = movie_ratings.notna()

    if valid_indices.sum() == 0:
        return user_means.get(user_id, train_data["user_rating"].mean())

    movie_ratings = movie_ratings[valid_indices]
    user_similarities = user_similarities[valid_indices]

    top_k_indices = user_similarities.nlargest(k).index
    top_k_similarities = user_similarities[top_k_indices]
    top_k_ratings = movie_ratings[top_k_indices]

    if top_k_similarities.sum() > 0:
        top_k_ratings_adjusted = top_k_ratings - user_means[top_k_indices]
        predicted_adjusted = np.dot(top_k_similarities, top_k_ratings_adjusted) / top_k_similarities.sum()
        return user_means.get(user_id, train_data["user_rating"].mean()) + predicted_adjusted
    else:
        return user_means.get(user_id, train_data["user_rating"].mean())


# 영화 추천 함수
def recommend_movies(user_id: str, num_recommendations: int = 5):
    if user_id not in user_item_matrix.index:
        print(f"사용자 {user_id}는 훈련 데이터에 없습니다. 인기 있는 영화를 추천합니다.")
        movie_means = user_item_matrix.mean(axis=0)
        recommended_movies = movie_means.nlargest(num_recommendations).index.tolist()
        recommended_scores = movie_means.nlargest(num_recommendations).values.tolist()
        return list(zip(recommended_movies, recommended_scores))

    unseen_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
    predictions = [(movie_id, predict_rating_knn(user_id, movie_id)) for movie_id in unseen_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_movies_with_scores = predictions[:num_recommendations]
    return recommended_movies_with_scores


# 테스트 데이터에 대한 예측 및 실제 평점 수집
predictions: list[float] = []
true_ratings: list[float] = []

for idx, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    prediction = predict_rating_knn(row["user_id"], row["movie_id"], k=20)
    predictions.append(prediction)
    true_ratings.append(row["user_rating"])

# RMSE 계산
rmse = sqrt(mean_squared_error(true_ratings, predictions))
print(f"모델의 RMSE: {rmse}")

# 사용자 입력 및 추천
user_id = input("추천받을 사용자 ID를 입력하세요: ")
recommended_movies_with_scores = recommend_movies(user_id, num_recommendations=5)

# 추천 영화와 점수 출력
print(f"사용자 {user_id}에게 추천하는 영화와 예측 평점:")
for movie, score in recommended_movies_with_scores:
    print(f"영화 {movie}: 예측 평점 {score:.2f}")
