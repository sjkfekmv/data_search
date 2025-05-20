import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import pickle
import re
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data loading and preprocessing
def load_data():
    """
    Load Dataset from File
    """
    # 读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python',encoding='latin-1')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    # 读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python',encoding='latin-1')
    movies_orig = movies.values
    # 将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d{4})\)$')

    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) + 1 - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    
    movies['Genres'] = movies['Genres'].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    
    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}
    
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])
    
    movies['Title'] = movies['Title'].map(title_map)

    # 读取评分数据集
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python',encoding='latin-1')
    ratings = ratings.filter(regex='UserID|MovieID|Rating')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)
    
    # 将数据分成X和y两张表
    target_fields = ['Rating']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    
    features = features_pd.values
    targets_values = targets_pd.values
    
    # 创建电影ID到索引的映射
    movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
    
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig, movieid2idx

# 2. Create MovieLensDataset
class MovieLensDataset(Dataset):
    def __init__(self, features, targets, movies, title_count=15, poster_dir='./poster/'):
        self.features = features
        self.targets = targets
        self.movies = movies
        self.title_count = title_count
        self.poster_dir = poster_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Extract features
        user_id = int(self.features[idx, 0]) - 1  # 0-indexed
        movie_id = int(self.features[idx, 1])
        gender = int(self.features[idx, 2])
        age = int(self.features[idx, 3])
        job_id = int(self.features[idx, 4])
        
        # Find movie in the movies dataframe
        movie_idx = self.movies[self.movies['MovieID'] == movie_id].index[0]
        movie_categories = self.movies.iloc[movie_idx]['Genres']
        movie_titles = self.movies.iloc[movie_idx]['Title']
        
        rating = float(self.targets[idx, 0])
        
        # Convert to tensors
        gender_tensor = torch.tensor(gender, dtype=torch.long)
        age_tensor = torch.tensor(age, dtype=torch.long)
        job_id_tensor = torch.tensor(job_id, dtype=torch.long)
        user_id_tensor = torch.tensor(user_id, dtype=torch.long)
        movie_id_tensor = torch.tensor(movie_id - 1, dtype=torch.long)  # 0-indexed
        
        # Convert categories and titles to tensors
        movie_categories_tensor = torch.tensor(movie_categories, dtype=torch.long)
        movie_titles_tensor = torch.tensor(movie_titles, dtype=torch.long)
        
        # Load movie poster image
        poster_path = os.path.join(self.poster_dir, f"{movie_id}.jpg")
        try:
            poster_img = Image.open(poster_path).convert('RGB')
            poster_img = self.transform(poster_img)
        except (FileNotFoundError, IOError):
            # If poster not found, use a placeholder of zeros
            poster_img = torch.zeros(3, 224, 224)
        
        return {
            'user_id': user_id_tensor,
            'movie_id': movie_id_tensor,
            'gender': gender_tensor,
            'age': age_tensor,
            'job_id': job_id_tensor,
            'movie_categories': movie_categories_tensor,
            'movie_titles': movie_titles_tensor,
            'poster_img': poster_img,
            'rating': torch.tensor(rating, dtype=torch.float)
        }

# 3. Define model architecture
class MovieRecommendationModel(nn.Module):
    def __init__(self, n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles, embed_dim=32, title_dim=15):
        super(MovieRecommendationModel, self).__init__()
        # User embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.gender_embedding = nn.Embedding(n_genders, embed_dim // 2)
        self.age_embedding = nn.Embedding(n_ages, embed_dim // 2)
        self.job_embedding = nn.Embedding(n_jobs, embed_dim // 2)
        self.embed_dim = embed_dim
        # Movie embeddings
        self.movie_embedding = nn.Embedding(n_movies, embed_dim)
        self.category_embedding = nn.Embedding(n_categories, embed_dim)
        self.title_embedding = nn.Embedding(n_titles, embed_dim)
        
        # Poster image feature extractor
        self.poster_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        
        # CNN for titles
        self.title_conv = nn.ModuleList([
            nn.Conv1d(embed_dim, 8, kernel_size=k) for k in [2, 3, 4, 5]
        ])
        # Calculate title feature dimension (8 channels per conv, 4 convs)
        self.title_feature_dim = 8 * len(self.title_conv)  # 8 * 4 = 32
        self.title_fc = nn.Linear(self.title_feature_dim, embed_dim)  # Project to embed_dim
        
        # Cross-modal attention for movie and poster features
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4)
        
        # Fusion layers
        self.fusion_fc1 = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_fc2 = nn.Linear(embed_dim, embed_dim)
        
        # User feature layers
        self.user_fc = nn.Linear(embed_dim, embed_dim)
        self.gender_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.age_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.job_fc = nn.Linear(embed_dim // 2, embed_dim)
        
        # Movie feature layers
        self.movie_fc = nn.Linear(embed_dim, embed_dim)
        self.category_fc = nn.Linear(embed_dim, embed_dim)
        
        # Final layers
        self.user_final = nn.Linear(4 * embed_dim, embed_dim)
        self.movie_final = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, user_id, movie_id, gender, age, job_id, movie_categories, movie_titles, poster_img):
        # User embeddings
        user_embed = self.user_embedding(user_id)  # [batch, embed_dim]
        gender_embed = self.gender_embedding(gender)  # [batch, embed_dim//2]
        age_embed = self.age_embedding(age)  # [batch, embed_dim//2]
        job_embed = self.job_embedding(job_id)  # [batch, embed_dim//2]
        
        # Movie embeddings
        movie_embed = self.movie_embedding(movie_id)  # [batch, embed_dim]
        
        # Process categories
        batch_size = movie_categories.size(0)
        category_embeds = []
        for i in range(batch_size):
            categories = movie_categories[i]
            category_embed = self.category_embedding(categories)  # [num_categories, embed_dim]
            category_embed = torch.sum(category_embed, dim=0, keepdim=True)  # [1, embed_dim]
            category_embeds.append(category_embed)
        category_embed = torch.cat(category_embeds, dim=0)  # [batch, embed_dim]
        
        # Process titles using CNN
        title_embed = self.title_embedding(movie_titles)  # [batch, seq_len, embed_dim]
        title_embed = title_embed.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        conv_outputs = []
        for conv in self.title_conv:
            x = F.relu(conv(title_embed))  # [batch, 8, seq_len-k+1]
            x = F.max_pool1d(x, x.size(2))  # [batch, 8, 1]
            conv_outputs.append(x)
        title_features = torch.cat(conv_outputs, dim=1)  # [batch, title_feature_dim]
        title_features = title_features.view(-1, self.title_feature_dim)  # [batch, 32]
        title_features = F.relu(self.title_fc(title_features))  # [batch, embed_dim]
        
        # Process movie poster
        poster_features = self.poster_encoder(poster_img)  # [batch, embed_dim]
        
        # Combine movie features (before attention)
        movie_combined = movie_embed + category_embed + title_features  # [batch, embed_dim]
        
        # Cross-modal attention
        movie_combined = movie_combined.unsqueeze(0)  # [1, batch, embed_dim]
        poster_features = poster_features.unsqueeze(0)  # [1, batch, embed_dim]
        attn_output, _ = self.cross_attention(movie_combined, poster_features, poster_features)
        attn_output = attn_output.squeeze(0)  # [batch, embed_dim]
        
        # Feature fusion
        fusion_input = torch.cat([attn_output, poster_features.squeeze(0)], dim=1)  # [batch, embed_dim*2]
        fusion_output = F.relu(self.fusion_fc1(fusion_input))  # [batch, embed_dim]
        fusion_output = F.relu(self.fusion_fc2(fusion_output))  # [batch, embed_dim]
        fusion_output = self.dropout(fusion_output)
        
        # User fully connected layers
        user_fc = F.relu(self.user_fc(user_embed))  # [batch, embed_dim]
        gender_fc = F.relu(self.gender_fc(gender_embed))  # [batch, embed_dim]
        age_fc = F.relu(self.age_fc(age_embed))  # [batch, embed_dim]
        job_fc = F.relu(self.job_fc(job_embed))  # [batch, embed_dim]
        
        # Combine user features
        user_combined = torch.cat([user_fc, gender_fc, age_fc, job_fc], dim=1)  # [batch, embed_dim*4]
        user_features = F.tanh(self.user_final(user_combined))  # [batch, embed_dim]
        user_features = self.dropout(user_features)
        
        # Final movie features
        movie_features = F.tanh(self.movie_final(fusion_output))  # [batch, embed_dim]
        
        # Final prediction
        prediction = torch.sum(user_features * movie_features, dim=1, keepdim=True)
        
        return prediction

# 4. Evaluation Metrics for Rating Prediction
class RatingMetrics:
    @staticmethod
    def mae(predictions, targets):
        """
        Mean Absolute Error
        """
        return torch.abs(predictions - targets).mean().item()
    
    @staticmethod
    def rmse(predictions, targets):
        """
        Root Mean Square Error
        """
        return torch.sqrt(((predictions - targets) ** 2).mean()).item()
    
    @staticmethod
    def nmae(predictions, targets, min_rating=1, max_rating=5):
        """
        Normalized Mean Absolute Error
        """
        range_ratings = max_rating - min_rating
        return torch.abs(predictions - targets).mean().item() / range_ratings
    
    @staticmethod
    def coverage(all_possible_items, recommended_items):
        """
        Coverage - percentage of items that the system is able to recommend
        """
        return len(recommended_items) / len(all_possible_items) * 100

# 5. Evaluation Metrics for Recommendation Lists
class RecommendationMetrics:
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=10):
        """
        Precision@k - Proportion of recommended items that are relevant
        """
        if len(recommended_items) == 0:
            return 0.0
            
        count = 0
        for item in recommended_items[:k]:
            if item in relevant_items:
                count += 1
        return count / min(k, len(recommended_items))
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        """
        Recall@k - Proportion of relevant items that are recommended
        """
        if len(relevant_items) == 0:
            return 0.0
            
        count = 0
        for item in recommended_items[:k]:
            if item in relevant_items:
                count += 1
        return count / len(relevant_items)
    
    @staticmethod
    def coverage(all_possible_items, recommended_items):
        """
        Coverage - percentage of items that the system is able to recommend
        """
        return len(recommended_items) / len(all_possible_items) * 100
    
    @staticmethod
    def calculate_auc(model, test_loader):
        """
        Calculate AUC based on predictions vs actual ratings
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Forward pass
                prediction = model(
                    batch['user_id'].to(device),
                    batch['movie_id'].to(device),
                    batch['gender'].to(device),
                    batch['age'].to(device),
                    batch['job_id'].to(device),
                    batch['movie_categories'].to(device),
                    batch['movie_titles'].to(device),
                    batch['poster_img'].to(device)
                )
                
                # Convert to binary for ROC (rating >= 4 is positive)
                pred_binary = (prediction >= 4.0).cpu().numpy()
                target_binary = (batch['rating'].view(-1, 1) >= 4.0).cpu().numpy()
                
                all_predictions.extend(pred_binary)
                all_targets.extend(target_binary)
        
        # Calculate metrics
        try:
            precision = precision_score(all_targets, all_predictions)
            recall = recall_score(all_targets, all_predictions)
            fpr, tpr, _ = roc_curve(all_targets, all_predictions)
            auc_score = auc(fpr, tpr)
            
            return precision, recall, auc_score
        except:
            return 0, 0, 0  # If there's an error due to insufficient data

# 6. Training function with evaluation
def train_model(train_loader, model, criterion, optimizer, val_loader=None, epochs=5):
    model.train()
    
    # Store metrics for each epoch
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_nmae': []
    }
    
    for epoch in range(epochs):
        # Training
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(
                batch['user_id'].to(device),
                batch['movie_id'].to(device),
                batch['gender'].to(device),
                batch['age'].to(device),
                batch['job_id'].to(device),
                batch['movie_categories'].to(device),
                batch['movie_titles'].to(device),
                batch['poster_img'].to(device)
            )
            
            # Compute loss
            loss = criterion(prediction, batch['rating'].view(-1, 1).to(device))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader:
            val_loss, val_mae, val_rmse, val_nmae = evaluate_model(val_loader, model, criterion)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            history['val_nmae'].append(val_nmae)
        
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, NMAE: {val_nmae:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
    return history

def evaluate_model(data_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            prediction = model(
                batch['user_id'].to(device),
                batch['movie_id'].to(device),
                batch['gender'].to(device),
                batch['age'].to(device),
                batch['job_id'].to(device),
                batch['movie_categories'].to(device),
                batch['movie_titles'].to(device),
                batch['poster_img'].to(device)
            )
            
            # Compute loss
            loss = criterion(prediction, batch['rating'].view(-1, 1).to(device))
            running_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.append(prediction.cpu())
            all_targets.append(batch['rating'].view(-1, 1))
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    mae = RatingMetrics.mae(all_predictions, all_targets)
    rmse = RatingMetrics.rmse(all_predictions, all_targets)
    nmae = RatingMetrics.nmae(all_predictions, all_targets)
    
    return running_loss / len(data_loader), mae, rmse, nmae

# 7. Generate feature matrices
# def generate_movie_features(model, movies, movieid2idx, poster_dir='./poster/'):
#     model.eval()
#     movie_features = []
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     with torch.no_grad():
#         for idx, item in tqdm(enumerate(movies.values), desc="Generating movie features"):
#             movie_id = item[0]
#             categories = torch.tensor(item[2], dtype=torch.long).to(device)
#             titles = torch.tensor(item[1], dtype=torch.long).to(device)
            
#             # Load poster image
#             poster_path = os.path.join(poster_dir, f"{int(movie_id)}.jpg")
#             try:
#                 poster_img = Image.open(poster_path).convert('RGB')
#                 poster_img = transform(poster_img)
#             except (FileNotFoundError, IOError):
#                 poster_img = torch.zeros(3, 224, 224)
            
#             poster_img = poster_img.unsqueeze(0).to(device)
            
#             # Get movie embedding
#             movie_embed = model.movie_embedding(torch.tensor([idx], dtype=torch.long).to(device)).view(-1, 1, 32)
            
#             # Process categories
#             category_embed = model.category_embedding(categories)
#             category_embed = torch.sum(category_embed, dim=0, keepdim=True).unsqueeze(0)
            
#             # Process titles
#             title_embed = model.title_embedding(titles.unsqueeze(0))
#             title_embed = title_embed.permute(0, 2, 1)
            
#             conv_outputs = []
#             for conv in model.title_conv:
#                 x = F.relu(conv(title_embed))
#                 x = F.max_pool1d(x, x.size(2))
#                 conv_outputs.append(x)
            
#             title_features = torch.cat(conv_outputs, dim=1)
#             title_features = title_features.view(-1, 1, 32)
            
#             # Process poster
#             # poster_features = model.poster_encoder(poster_img).view(-1, 1, 200)
#             poster_features = model.poster_encoder(poster_img)
#             # Movie fully connected layers
#             movie_fc = F.relu(model.movie_fc(movie_embed))
#             category_fc = F.relu(model.category_fc(category_embed))
            
#             # Combine movie features
#             movie_combined = torch.cat([movie_fc, category_fc, title_features], dim=2)
#             movie_features_vec = F.tanh(model.movie_final(movie_combined))
#             # movie_features_vec = movie_features_vec.view(-1, 200)
            
#             # Add poster features
#             # poster_features = poster_features.view(-1, 200)
#             movie_features_vec = movie_features_vec + poster_features
            
#             movie_features.append(movie_features_vec.cpu().numpy())
    
#     return np.array(movie_features).reshape(-1, 200)


def generate_movie_features(model, movies, movieid2idx, poster_dir='./poster/'):
    model.eval()
    movie_features = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for idx, item in tqdm(enumerate(movies.values), desc="Generating movie features"):
            movie_id = item[0]
            categories = torch.tensor(item[2], dtype=torch.long).to(device)
            titles = torch.tensor(item[1], dtype=torch.long).to(device)
            
            # Load poster image
            poster_path = os.path.join(poster_dir, f"{int(movie_id)}.jpg")
            try:
                poster_img = Image.open(poster_path).convert('RGB')
                poster_img = transform(poster_img)
            except (FileNotFoundError, IOError):
                poster_img = torch.zeros(3, 224, 224)
            
            poster_img = poster_img.unsqueeze(0).to(device)
            
            # Get movie embedding
            movie_embed = model.movie_embedding(torch.tensor([idx], dtype=torch.long).to(device))  # [1, embed_dim]
            
            # Process categories
            category_embed = model.category_embedding(categories)  # [num_categories, embed_dim]
            category_embed = torch.sum(category_embed, dim=0, keepdim=True)  # [1, embed_dim]
            
            # Process titles
            title_embed = model.title_embedding(titles.unsqueeze(0))  # [1, seq_len, embed_dim]
            title_embed = title_embed.permute(0, 2, 1)  # [1, embed_dim, seq_len]
            
            conv_outputs = []
            for conv in model.title_conv:
                x = F.relu(conv(title_embed))  # [1, 8, seq_len-k+1]
                x = F.max_pool1d(x, x.size(2))  # [1, 8, 1]
                conv_outputs.append(x)
            
            title_features = torch.cat(conv_outputs, dim=1)  # [1, title_feature_dim]
            title_features = title_features.view(1, model.title_feature_dim)  # [1, title_feature_dim]
            title_features = F.relu(model.title_fc(title_features))  # [1, embed_dim]
            
            # Process poster
            poster_features = model.poster_encoder(poster_img)  # [1, embed_dim]
            
            # Combine movie features
            movie_combined = movie_embed + category_embed + title_features  # [1, embed_dim]
            
            # Movie fully connected layers
            movie_features_vec = F.tanh(model.movie_final(movie_combined))  # [1, embed_dim]
            
            # Add poster features
            movie_features_vec = movie_features_vec + poster_features  # [1, embed_dim]
            
            movie_features.append(movie_features_vec.cpu().numpy())
    
    return np.array(movie_features).reshape(-1, model.embed_dim)  # 动态使用 embed_dim


def generate_user_features(model, users):
    model.eval()
    user_features = []
    
    # 确保模型在正确设备上
    model_device = next(model.parameters()).device
    if device.type == "cuda":
        # 允许 cuda 和 cuda:0 等价
        assert model_device.type == "cuda", f"Model is on {model_device}, but expected a CUDA device"
    else:
        assert model_device == device, f"Model is on {model_device}, but expected {device}"
    
    with torch.no_grad():
        for idx, item in tqdm(enumerate(users.values), desc="Generating user features"):
            try:
                # 确保输入数据为整数并移动到正确设备
                user_id = torch.tensor([idx], dtype=torch.long).to(device)
                gender = torch.tensor([int(item[1])], dtype=torch.long).to(device)
                age = torch.tensor([int(item[2])], dtype=torch.long).to(device)
                job_id = torch.tensor([int(item[3])], dtype=torch.long).to(device)
                
                # User embeddings
                user_embed = model.user_embedding(user_id).view(-1, 1, model.embed_dim)  # [1, 1, embed_dim]
                gender_embed = model.gender_embedding(gender).view(-1, 1, model.embed_dim // 2)  # [1, 1, embed_dim//2]
                age_embed = model.age_embedding(age).view(-1, 1, model.embed_dim // 2)  # [1, 1, embed_dim//2]
                job_embed = model.job_embedding(job_id).view(-1, 1, model.embed_dim // 2)  # [1, 1, embed_dim//2]
                
                # User fully connected layers
                user_fc = F.relu(model.user_fc(user_embed))  # [1, 1, embed_dim]
                gender_fc = F.relu(model.gender_fc(gender_embed))  # [1, 1, embed_dim]
                age_fc = F.relu(model.age_fc(age_embed))  # [1, 1, embed_dim]
                job_fc = F.relu(model.job_fc(job_embed))  # [1, 1, embed_dim]
                
                # Combine user features
                user_combined = torch.cat([user_fc, gender_fc, age_fc, job_fc], dim=2)  # [1, 1, 4*embed_dim]
                user_features_vec = F.tanh(model.user_final(user_combined))  # [1, 1, embed_dim]
                
                # 使用动态维度
                user_features_vec = user_features_vec.view(-1, model.embed_dim)  # [1, embed_dim]
                
                user_features.append(user_features_vec.cpu().numpy())
            except (ValueError, TypeError) as e:
                print(f"Error processing user {idx}: {e}")
                continue
    
    return np.array(user_features).reshape(-1, model.embed_dim)  # [n_users, embed_dim]
# 8. Movie recommendation functions with evaluation
def recommend_same_type_movie(model, movie_id, movie_matrics, movies_orig, movieid2idx, actual_ratings=None, top_k=20):
    model.eval()
    movie_features = torch.tensor(movie_matrics, dtype=torch.float).to(device)
    probs_embeddings = torch.tensor(movie_matrics[movieid2idx[movie_id]], dtype=torch.float).view(1, -1).to(device)
    
    # Compute similarity
    norm_movie_matrics = torch.sqrt(torch.sum(torch.square(movie_features), dim=1, keepdim=True))
    normalized_movie_matrics = movie_features / norm_movie_matrics
    probs_similarity = torch.matmul(probs_embeddings, normalized_movie_matrics.t())
    
    # Convert to numpy for further processing
    sim = probs_similarity.cpu().numpy()
    
    print(f"您看的电影是：{movies_orig[movieid2idx[movie_id]]}")
    print("以下是给您的推荐：")
    
    p = np.squeeze(sim)
    
    # Get top k similar movies
    top_indices = np.argsort(p)[-top_k:][::-1]  # Get top k items
    
    # Sample from top k for diversity
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(len(movie_matrics), 1, p=p)[0]
        results.add(c)
    
    for val in results:
        print(val)
        print(movies_orig[val])
    
    # Evaluate metrics
    if actual_ratings is not None:
        # Create a set of movies the user actually liked (rated >= 4)
        actual_liked_movies = set([i for i, movie_id in enumerate(movies_orig[:, 0]) 
                                if movie_id in actual_ratings and actual_ratings[movie_id] >= 4])
        
        # Calculate precision and recall
        recommended_items = set(top_indices[:10])  # Use top 10 for metrics
        precision = RecommendationMetrics.precision_at_k(list(top_indices), actual_liked_movies, 10)
        recall = RecommendationMetrics.recall_at_k(list(top_indices), actual_liked_movies, 10)
        
        print(f"Precision@10: {precision:.4f}")
        print(f"Recall@10: {recall:.4f}")
    
    # Evaluate coverage
    all_movies = set(range(len(movies_orig)))
    recommended_items = set(top_indices[:10])
    coverage = RatingMetrics.coverage(all_movies, recommended_items)
    print(f"Coverage: {coverage:.2f}%")
    
    return results

def recommend_your_favorite_movie(model, user_id, users_matrics, movie_matrics, movies_orig, 
                                  actual_user_ratings=None, k=10):
    """
    Recommends movies to a user and evaluates the recommendations
    """
    model.eval()
    user_features = torch.tensor(users_matrics[user_id-1], dtype=torch.float).view(1, -1).to(device)
    movie_features = torch.tensor(movie_matrics, dtype=torch.float).to(device)
    
    # Compute predictions
    probs_similarity = torch.matmul(user_features, movie_features.t())
    sim = probs_similarity.cpu().numpy()
    
    print("以下是给您的推荐：")
    
    # Get top recommendations
    p = np.squeeze(sim)
    top_indices = np.argsort(p)[-k:][::-1]  # Top k items
    results = set()
    
    # Random sample from top items (for diversity)
    while len(results) != 5:
        c = np.random.choice(top_indices, 1)[0]
        results.add(c)
    
    for val in results:
        print(val)
        print(movies_orig[val])
    
    # If we have actual ratings, evaluate recommendation quality
    if actual_user_ratings is not None:
        # Get relevant items (items the user likes, e.g., rated >= 4)
        relevant_items = set([i for i, r in enumerate(actual_user_ratings) if r >= 4.0])
        
        # Calculate precision and recall
        precision = RecommendationMetrics.precision_at_k(list(top_indices), relevant_items, k)
        recall = RecommendationMetrics.recall_at_k(list(top_indices), relevant_items, k)
        
        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}: {recall:.4f}")
    
    return results

def recommend_other_favorite_movie(model, movie_id, users_matrics, movie_matrics, movies_orig, 
                                  users_orig, movieid2idx, actual_ratings=None, top_k=20):
    """
    Recommends movies based on what users who like the given movie also like
    """
    model.eval()
    movie_features = torch.tensor(movie_matrics[movieid2idx[movie_id]], dtype=torch.float).view(1, -1).to(device)
    user_features = torch.tensor(users_matrics, dtype=torch.float).to(device)
    
    # Find users who like this movie
    probs_user_favorite_similarity = torch.matmul(movie_features, user_features.t())
    favorite_user_ids = torch.argsort(probs_user_favorite_similarity.cpu().squeeze())[-top_k:].numpy()
    
    print(f"您看的电影是：{movies_orig[movieid2idx[movie_id]]}")
    print(f"喜欢看这个电影的人是：{users_orig[favorite_user_ids]}")
    
    # Get favorite movies for those users
    selected_users = torch.tensor(users_matrics[favorite_user_ids], dtype=torch.float).to(device)
    selected_movies = torch.tensor(movie_matrics, dtype=torch.float).to(device)
    
    # Find what movies these users like
    probs_similarity = torch.matmul(selected_users, selected_movies.t())
    
    # Get top k recommendations across all users
    # First, compute the average similarity score across all selected users
    avg_sim = torch.mean(probs_similarity, dim=0)
    # Get indices of top k movies
    top_movie_indices = torch.argsort(avg_sim, descending=True)[:top_k].cpu().numpy()
    
    # Also get individual user preferences for evaluation
    liked_movies = torch.argmax(probs_similarity, dim=1).cpu().numpy()
    
    print("喜欢看这个电影的人还喜欢看：")
    
    # Select 5 movies to recommend (either from the top k or from individual user preferences)
    if len(set(top_movie_indices)) < 5:
        results = set(top_movie_indices)
        # Add some from individual preferences if needed
        remaining = 5 - len(results)
        if remaining > 0 and len(set(liked_movies)) > 0:
            for movie in liked_movies:
                if movie not in results:
                    results.add(movie)
                    remaining -= 1
                    if remaining == 0:
                        break
    else:
        # Sample from top movies
        results = set()
        while len(results) != 5:
            c = top_movie_indices[random.randrange(min(10, len(top_movie_indices)))]
            if c != movieid2idx[movie_id]:  # Avoid recommending the same movie
                results.add(c)
    
    for val in results:
        print(val)
        print(movies_orig[val])
    
    # Calculate evaluation metrics for the recommendations
    if actual_ratings is not None:
        # Define relevant items as those with ratings >= 4
        all_movies = set(range(len(movies_orig)))
        actual_liked_movies = set([i for i, movie_id in enumerate(movies_orig[:, 0]) 
                                if movie_id in actual_ratings and actual_ratings[movie_id] >= 4])
        
        # Calculate precision and recall
        recommended_items = set(top_movie_indices[:10])
        precision = RecommendationMetrics.precision_at_k(list(top_movie_indices), actual_liked_movies, 10)
        recall = RecommendationMetrics.recall_at_k(list(top_movie_indices), actual_liked_movies, 10)
        
        print(f"Precision@10: {precision:.4f}")
        print(f"Recall@10: {recall:.4f}")
        
        # Calculate coverage
        coverage = RecommendationMetrics.coverage(all_movies, recommended_items)
        print(f"Coverage: {coverage:.2f}%")
    
    return results


def main():
    # Load data
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig, movieid2idx = load_data()
    
    # Create dataset and dataloader
    dataset = MovieLensDataset(features, targets_values, movies, title_count)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Initialize model
    n_users = int(np.max(users.values[:, 0]))  # 0-indexed
    n_movies = int(np.max(movies.values[:, 0]))  # 0-indexed
    n_genders = 2  # 0: Female, 1: Male
    n_ages = len(set(users['Age']))
    n_jobs = len(set(users['JobID'])) + 1  # +1 for potential unknown jobs
    n_categories = max(max(genres2int.values()), 
                        np.max([max(x) if len(x) > 0 else 0 for x in movies['Genres']])) + 1
    n_titles = len(title_set)
    
    print(f"n_users: {n_users}, n_movies: {n_movies}, n_categories: {n_categories}, n_titles: {n_titles}")
    
    model = MovieRecommendationModel(
        n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_model(train_loader, model, criterion, optimizer, epochs=5)
    
    # Save model
    torch.save(model.state_dict(), 'movie_recommendation_model.pth')
    
    # Generate features
    movie_matrics = generate_movie_features(model, movies, movieid2idx)
    users_matrics = generate_user_features(model, users)
    
    # Save features
    pickle.dump(movie_matrics, open('movie_matrics.p', 'wb'))
    pickle.dump(users_matrics, open('users_matrics.p', 'wb'))
    
    # Example recommendations
    print("\n--- Recommendations for similar movies ---")
    recommend_same_type_movie(model, 1401, movie_matrics, movies_orig, movieid2idx)
    
    print("\n--- Recommendations for user ---")
    recommend_your_favorite_movie(model, 234, users_matrics, movie_matrics, movies_orig)
    
    print("\n--- Recommendations based on other users ---")
    recommend_other_favorite_movie(model, 1401, users_matrics, movie_matrics, movies_orig, users_orig, movieid2idx)

if __name__ == "__main__":
    main()