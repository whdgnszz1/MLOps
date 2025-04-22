import os

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import opendatasets as od
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

dataset = "https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"
od.download(dataset)


def load_netflix_data(data_dir):
    files = [os.path.join(data_dir, f) for f in
             ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]]
    data = pd.DataFrame()

    for file in files:
        df = pd.read_csv(file, header=None, names=["Cust_Id", "Rating"], usecols=[0, 1])
        df["Rating"] = df["Rating"].astype(float)
        df["Movie_Id"] = df["Cust_Id"].apply(lambda x: x[:-1] if ":" in x else np.nan).ffill().astype(int)
        df = df[df["Rating"].notna()]
        data = pd.concat([data, df])

    data["Cust_Id"] = data["Cust_Id"].astype(int)

    unique_users = data["Cust_Id"].unique()
    unique_movies = data["Movie_Id"].unique()
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}

    data["User_Idx"] = data["Cust_Id"].map(user_to_idx)
    data["Movie_Idx"] = data["Movie_Id"].map(movie_to_idx)

    return data, len(unique_users), len(unique_movies)


class NetflixDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data["User_Idx"].values, dtype=torch.long)
        self.movies = torch.tensor(data["Movie_Idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(data["Rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


class RecommenderNN(nn.Module):
    def __init__(self, num_users, num_movies, emb_size=100):
        super(RecommenderNN, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.movie_emb = nn.Embedding(num_movies, emb_size)
        self.fc1 = nn.Linear(2 * emb_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user, movie):
        user_emb = self.user_emb(user)
        movie_emb = self.movie_emb(movie)
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


# 데이터 로드
data_dir = "netflix-prize-data"
data, num_users, num_movies = load_netflix_data(data_dir)

# 데이터셋 및 데이터로더 생성
dataset = NetflixDataset(data)
data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

# 모델 초기화
model = RecommenderNN(num_users=num_users, num_movies=num_movies)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(5):
    model.train()
    total_loss = 0
    for users, movies, ratings in tqdm(data_loader):
        optimizer.zero_grad()
        outputs = model(users, movies)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(data_loader)}")