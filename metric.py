import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pickle
import re
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model architecture (exactly the same as in training)
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

# Dataset class (same as in training)
# class MovieLensDataset(Dataset):
#     def __init__(self, features, targets, movies, title_count=15, poster_dir='./poster/'):
#         self.features = features
#         self.targets = targets
#         self.movies = movies
#         self.title_count = title_count
#         self.poster_dir = poster_dir
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
        
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         # Extract features
#         user_id = int(self.features[idx, 0]) - 1  # 0-indexed
#         movie_id = int(self.features[idx, 1])
#         gender = int(self.features[idx, 2])
#         age = int(self.features[idx, 3])
#         job_id = int(self.features[idx, 4])
        
#         # Find movie in the movies dataframe
#         movie_idx = self.movies[self.movies['MovieID'] == movie_id].index[0]
#         movie_categories = self.movies.iloc[movie_idx]['Genres']
#         movie_titles = self.movies.iloc[movie_idx]['Title']
        
#         rating = float(self.targets[idx, 0])
        
#         # Convert to tensors
#         gender_tensor = torch.tensor(gender, dtype=torch.long)
#         age_tensor = torch.tensor(age, dtype=torch.long)
#         job_id_tensor = torch.tensor(job_id, dtype=torch.long)
#         user_id_tensor = torch.tensor(user_id, dtype=torch.long)
#         movie_id_tensor = torch.tensor(movie_id - 1, dtype=torch.long)  # 0-indexed
        
#         # Convert categories and titles to tensors
#         movie_categories_tensor = torch.tensor(movie_categories, dtype=torch.long)
#         movie_titles_tensor = torch.tensor(movie_titles, dtype=torch.long)
        
#         # Load movie poster image
#         poster_path = os.path.join(self.poster_dir, f"{movie_id}.jpg")
#         try:
#             poster_img = Image.open(poster_path).convert('RGB')
#             poster_img = self.transform(poster_img)
#         except (FileNotFoundError, IOError):
#             # If poster not found, use a placeholder of zeros
#             poster_img = torch.zeros(3, 224, 224)
#         basic=True
#         if basic:
#             poster_img = torch.zeros(3, 224, 224)

#         return {
#             'user_id': user_id_tensor,
#             'movie_id': movie_id_tensor,
#             'gender': gender_tensor,
#             'age': age_tensor,
#             'job_id': job_id_tensor,
#             'movie_categories': movie_categories_tensor,
#             'movie_titles': movie_titles_tensor,
#             'poster_img': poster_img,
#             'rating': torch.tensor(rating, dtype=torch.float)
#         }
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
        
        # Handle 1D or 2D targets
        if self.targets.ndim == 1:
            rating = float(self.targets[idx])
        else:
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
            poster_img = torch.zeros(3, 224, 224)
        basic = True
        if basic:
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
# Data loading function
def load_data():
    # Same as in training code
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python',encoding='latin-1')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python',encoding='latin-1')
    movies_orig = movies.values
    
    pattern = re.compile(r'^(.*)\((\d{4})\)$')
    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) + 1 - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    
    movies['Genres'] = movies['Genres'].map(genres_map)

    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    
    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}
    
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])
    
    movies['Title'] = movies['Title'].map(title_map)

    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python',encoding='latin-1')
    ratings = ratings.filter(regex='UserID|MovieID|Rating')

    data = pd.merge(pd.merge(ratings, users), movies)
    
    target_fields = ['Rating']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    
    features = features_pd.values
    targets_values = targets_pd.values
    
    movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
    
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig, movieid2idx

# Evaluation Metrics
class RatingMetrics:
    @staticmethod
    def mae(predictions, targets):
        """Mean Absolute Error"""
        return torch.abs(predictions - targets).mean().item()
    
    @staticmethod
    def rmse(predictions, targets):
        """Root Mean Square Error"""
        return torch.sqrt(((predictions - targets) ** 2).mean()).item()
    
    @staticmethod
    def nmae(predictions, targets, min_rating=1, max_rating=5):
        """Normalized Mean Absolute Error"""
        range_ratings = max_rating - min_rating
        return torch.abs(predictions - targets).mean().item() / range_ratings
    
    @staticmethod
    def coverage(all_possible_items, recommended_items):
        """Coverage - percentage of items that the system is able to recommend"""
        return len(recommended_items) / len(all_possible_items) * 100

class RecommendationMetrics:
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=10):
        """Precision@k - Proportion of recommended items that are relevant"""
        if len(recommended_items) == 0:
            return 0.0
            
        count = 0
        for item in recommended_items[:k]:
            if item in relevant_items:
                count += 1
        return count / min(k, len(recommended_items))
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        """Recall@k - Proportion of relevant items that are recommended"""
        if len(relevant_items) == 0:
            return 0.0
            
        count = 0
        for item in recommended_items[:k]:
            if item in relevant_items:
                count += 1
        return count / len(relevant_items)

# Evaluate model function
def evaluate_model(data_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating model"):
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
    
    return running_loss / len(data_loader), mae, rmse, nmae, all_predictions.numpy(), all_targets.numpy()

# Calculate ROC and AUC
def calculate_roc_and_auc(predictions, targets, threshold=2.0):
    # Convert to binary (rating >= threshold is positive)
    pred_binary = (predictions >= threshold).astype(int)
    target_binary = (targets >= threshold).astype(int)
    
    # Calculate ROC curve and AUC
    try:
        fpr, tpr, _ = roc_curve(target_binary, predictions)
        auc_score = auc(fpr, tpr)
        
        # Calculate precision and recall (for the binary classification)
        precision = precision_score(target_binary, pred_binary)
        recall = recall_score(target_binary, pred_binary)
        
        return fpr, tpr, auc_score, precision, recall
    except:
        print("Error calculating ROC/AUC. Check if there are enough positive/negative samples.")
        return None, None, 0, 0, 0

# Recommendation functions
def recommend_movies_for_user(model, user_id, users_matrics, movie_matrics, k=10):
    user_features = torch.tensor(users_matrics[user_id-1], dtype=torch.float).view(1, -1).to(device)
    movie_features = torch.tensor(movie_matrics, dtype=torch.float).to(device)
    
    # Compute similarity/prediction scores
    scores = torch.matmul(user_features, movie_features.t())
    
    # Get top-k movie indices
    p = scores.cpu().numpy().squeeze()
    top_indices = np.argsort(p)[-k:][::-1]
    
    return top_indices

def get_actual_liked_movies(user_id, data, threshold=4.0):
    # Get movies this user has rated >= threshold
    user_ratings = data[data['UserID'] == user_id]
    liked_movies = user_ratings[user_ratings['Rating'] >= threshold]['MovieID'].values
    return liked_movies

def calculate_coverage(model, users_matrics, movie_matrics, n_users, n_movies, sample_size=100):
    # Sample users to evaluate coverage
    sampled_users = np.random.choice(n_users, min(sample_size, n_users), replace=False)
    
    # Track all recommended movies
    all_recommended = set()
    
    for user_id in tqdm(sampled_users, desc="Calculating coverage"):
        user_id = int(user_id) + 1  # Convert to 1-indexed
        recommendations = recommend_movies_for_user(model, user_id, users_matrics, movie_matrics, k=10)
        all_recommended.update(recommendations)
    
    # Calculate coverage
    all_possible = set(range(n_movies))
    coverage = RatingMetrics.coverage(all_possible, all_recommended)
    
    return coverage
def evaluate_recommendation_system(model_path='movie_recommendation_model.pth'):
    print("Loading data...")
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig, movieid2idx = load_data()
    
    # 按用户划分测试集
    print("Splitting data by users...")
    min_ratings = 10
    user_rating_counts = ratings.groupby('UserID').size()
    valid_users = user_rating_counts[user_rating_counts >= min_ratings].index
    valid_ratings = ratings[ratings['UserID'].isin(valid_users)]
    
    train_users, test_users = train_test_split(valid_users, test_size=0.2, random_state=42)
    train_mask = valid_ratings['UserID'].isin(train_users)
    test_mask = valid_ratings['UserID'].isin(test_users)
    train_ratings = valid_ratings[train_mask]
    test_ratings = valid_ratings[test_mask]
    
    test_data = pd.merge(pd.merge(test_ratings, users), movies)
    train_data = pd.merge(pd.merge(train_ratings, users), movies)
    
    train_features = train_data.drop('Rating', axis=1).values
    test_features = test_data.drop('Rating', axis=1).values
    # train_targets = train_data['Rating'].values
    # test_targets = test_data['Rating'].values
    test_targets = test_data['Rating'].values.reshape(-1, 1)
    train_targets = train_data['Rating'].values.reshape(-1, 1)
    print(f"Test set size: {len(test_features)}")
    
    # Create test dataset and dataloader
    test_dataset = MovieLensDataset(test_features, test_targets, movies, title_count)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    n_users = int(np.max(users.values[:, 0]))
    n_movies = int(np.max(movies.values[:, 0]))
    n_genders = 2
    n_ages = len(set(users['Age']))
    n_jobs = len(set(users['JobID'])) + 1
    n_categories = max(max(genres2int.values()), 
                      np.max([max(x) if len(x) > 0 else 0 for x in movies['Genres']])) + 1
    n_titles = len(title_set)
    
    model = MovieRecommendationModel(
        n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles
    ).to(device)
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluate rating prediction
    print("\n=== Evaluating Rating Prediction Metrics ===")
    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_nmae, all_preds, all_targets = evaluate_model(test_loader, model, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"NMAE: {test_nmae:.4f}")
    
    # Calculate ROC and AUC
    print("\n=== Calculating ROC and AUC ===")
    fpr, tpr, auc_score, precision, recall = calculate_roc_and_auc(all_preds.flatten(), all_targets.flatten(), threshold=3.5)
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Binary Classification Precision: {precision:.4f}")
    print(f"Binary Classification Recall: {recall:.4f}")
    
    # Plot ROC curve
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
        print("ROC curve saved to 'roc_curve.png'")
    
    # Generate or load feature matrices
    try:
        print("\n=== Loading Feature Matrices ===")
        movie_matrics = pickle.load(open('movie_matrics.p', 'rb'))
        users_matrics = pickle.load(open('users_matrics.p', 'rb'))
        print("Feature matrices loaded successfully")
    except:
        print("\n=== Generating Feature Matrices (this may take a while) ===")
        from metricmain import generate_movie_features, generate_user_features
        movie_matrics = generate_movie_features(model, movies, movieid2idx)
        users_matrics = generate_user_features(model, users)
        pickle.dump(movie_matrics, open('movie_matrics.p', 'wb'))
        pickle.dump(users_matrics, open('users_matrics.p', 'wb'))
    
    # Calculate coverage
    print("\n=== Calculating System Coverage ===")
    coverage = calculate_coverage(model, users_matrics, movie_matrics, n_users, n_movies)
    print(f"Coverage: {coverage:.2f}%")
    
    # Evaluate recommendation quality
    print("\n=== Evaluating Recommendation Quality ===")
    min_liked_movies = 5
    valid_users = []
    for user_id in test_data['UserID'].unique():
        liked_movies, _ = get_actual_liked_movies(user_id, test_data)
        if len(liked_movies) >= min_liked_movies:
            valid_users.append(user_id)
    
    print(f"Found {len(valid_users)} valid users with at least {min_liked_movies} liked movies")
    sample_users = np.random.choice(valid_users, min(100, len(valid_users)), replace=False)
    
    avg_precision = 0
    avg_recall = 0
    user_count = 0
    
    for user_id in sample_users:
        liked_movies, threshold = get_actual_liked_movies(user_id, test_data)
        relevant_indices = [movieid2idx[movie_id] for movie_id in liked_movies if movie_id in movieid2idx]
        if len(relevant_indices) > 0:
            recommended_indices = recommend_movies_for_user(model, user_id, users_matrics, movie_matrics, ratings, movieid2idx)
            precision = RecommendationMetrics.precision_at_k(recommended_indices, set(relevant_indices))
            recall = RecommendationMetrics.recall_at_k(recommended_indices, set(relevant_indices))
            avg_precision += precision
            avg_recall += recall
            user_count += 1
            print(f"User {user_id}: {len(liked_movies)} liked movies (threshold={threshold:.2f}), Precision@10={precision:.4f}, Recall@10={recall:.4f}")
    
    if user_count > 0:
        avg_precision /= user_count
        avg_recall /= user_count
        print(f"\nAverage Precision@10: {avg_precision:.4f}")
        print(f"Average Recall@10: {avg_recall:.4f}")
    
    print("\n=== Final Evaluation Summary ===")
    print(f"Rating Prediction MAE: {test_mae:.4f}")
    print(f"Rating Prediction RMSE: {test_rmse:.4f}")
    print(f"Rating Prediction NMAE: {test_nmae:.4f}")
    print(f"Binary Classification AUC: {auc_score:.4f}")
    print(f"Binary Classification Precision: {precision:.4f}")
    print(f"Binary Classification Recall: {recall:.4f}")
    print(f"Recommendation Precision@10: {avg_precision:.4f}")
    print(f"Recommendation Recall@10: {avg_recall:.4f}")
    print(f"System Coverage: {coverage:.2f}%")
    print("\nRecommendation system evaluation complete!")
# Main evaluation function
# def evaluate_recommendation_system(model_path='movie_recommendation_model.pth'):
#     print("Loading data...")
#     title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig, movieid2idx = load_data()
    
#     # Split data for evaluation
#     _, test_features, _, test_targets = train_test_split(
#         features, targets_values, test_size=0.2, random_state=42)
    
#     print(f"Test set size: {len(test_features)}")
    
#     # Create test dataset and dataloader
#     test_dataset = MovieLensDataset(test_features, test_targets, movies, title_count)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
#     # Initialize model parameters
#     n_users = int(np.max(users.values[:, 0]))
#     n_movies = int(np.max(movies.values[:, 0]))
#     n_genders = 2  # 0: Female, 1: Male
#     n_ages = len(set(users['Age']))
#     n_jobs = len(set(users['JobID'])) + 1
#     n_categories = max(max(genres2int.values()), 
#                      np.max([max(x) if len(x) > 0 else 0 for x in movies['Genres']])) + 1
#     n_titles = len(title_set)
    
#     # Initialize model with the same architecture
#     model = MovieRecommendationModel(
#         n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles
#     ).to(device)
    
#     # Load pre-trained model weights
#     print(f"Loading model from {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # Define loss function for evaluation
#     criterion = nn.MSELoss()
    
#     # Evaluate rating prediction metrics
#     print("\n=== Evaluating Rating Prediction Metrics ===")
#     test_loss, test_mae, test_rmse, test_nmae, all_preds, all_targets = evaluate_model(test_loader, model, criterion)
#     print(f"Test Loss: {test_loss:.4f}")
#     print(f"MAE: {test_mae:.4f}")
#     print(f"RMSE: {test_rmse:.4f}")
#     print(f"NMAE: {test_nmae:.4f}")
    
#     # Calculate ROC and AUC
#     print("\n=== Calculating ROC and AUC ===")
#     fpr, tpr, auc_score, precision, recall = calculate_roc_and_auc(all_preds.flatten(), all_targets.flatten())
#     print(f"AUC Score: {auc_score:.4f}")
#     print(f"Binary Classification Precision: {precision:.4f}")
#     print(f"Binary Classification Recall: {recall:.4f}")
    
#     # Plot ROC curve if available
#     if fpr is not None and tpr is not None:
#         plt.figure(figsize=(10, 8))
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic (ROC) Curve')
#         plt.legend(loc="lower right")
#         plt.savefig('roc_curve.png')
#         print("ROC curve saved to 'roc_curve.png'")
    
#     # Try to load pre-computed features if available
#     try:
#         print("\n=== Loading Feature Matrices ===")
#         movie_matrics = pickle.load(open('movie_matrics.p', 'rb'))
#         users_matrics = pickle.load(open('users_matrics.p', 'wb'))
#         print("Feature matrices loaded successfully")
#     except:
#         # Generate feature matrices if not available
#         print("\n=== Generating Feature Matrices (this may take a while) ===")
#         print("This step is necessary for evaluating recommendation quality")
        
#         # Load feature matrix generation functions from your module
#         # Assuming these functions are in your model file
#         from metricmain import generate_movie_features, generate_user_features
        
#         movie_matrics = generate_movie_features(model, movies, movieid2idx)
#         users_matrics = generate_user_features(model, users)
        
#         # Save for future use
#         pickle.dump(movie_matrics, open('movie_matrics.p', 'wb'))
#         pickle.dump(users_matrics, open('users_matrics.p', 'wb'))
    
#     # Calculate coverage
#     print("\n=== Calculating System Coverage ===")
#     coverage = calculate_coverage(model, users_matrics, movie_matrics, n_users, n_movies)
#     print(f"Coverage: {coverage:.2f}%")
    
#     # Evaluate recommendation quality for a sample of users
#     print("\n=== Evaluating Recommendation Quality ===")
#     sample_users = np.random.choice(n_users, min(20, n_users), replace=False)
    
#     avg_precision = 0
#     avg_recall = 0
#     user_count = 0
    
#     for user_id in sample_users:
#         user_id = int(user_id) + 1  # Convert to 1-indexed
        
#         # Get user's actual liked movies
#         liked_movies = get_actual_liked_movies(user_id, data)
        
#         if len(liked_movies) > 0:
#             # Convert movie IDs to indices
#             relevant_indices = [movieid2idx[movie_id] for movie_id in liked_movies if movie_id in movieid2idx]
            
#             if len(relevant_indices) > 0:
#                 # Get recommendations for this user
#                 recommended_indices = recommend_movies_for_user(model, user_id, users_matrics, movie_matrics, k=10)
                
#                 # Calculate precision and recall
#                 precision = RecommendationMetrics.precision_at_k(recommended_indices, set(relevant_indices))
#                 recall = RecommendationMetrics.recall_at_k(recommended_indices, set(relevant_indices))
                
#                 avg_precision += precision
#                 avg_recall += recall
#                 user_count += 1
                
#                 print(f"User {user_id}: Precision@10={precision:.4f}, Recall@10={recall:.4f}")
    
#     if user_count > 0:
#         avg_precision /= user_count
#         avg_recall /= user_count
#         print(f"\nAverage Precision@10: {avg_precision:.4f}")
#         print(f"Average Recall@10: {avg_recall:.4f}")
    
#     # Print summary of all metrics
#     print("\n=== Final Evaluation Summary ===")
#     print(f"Rating Prediction MAE: {test_mae:.4f}")
#     print(f"Rating Prediction RMSE: {test_rmse:.4f}")
#     print(f"Rating Prediction NMAE: {test_nmae:.4f}")
#     print(f"Binary Classification AUC: {auc_score:.4f}")
#     print(f"Binary Classification Precision: {precision:.4f}")
#     print(f"Binary Classification Recall: {recall:.4f}")
#     print(f"Recommendation Precision@10: {avg_precision:.4f}")
#     print(f"Recommendation Recall@10: {avg_recall:.4f}")
#     print(f"System Coverage: {coverage:.2f}%")
    
#     print("\nRecommendation system evaluation complete!")

if __name__ == "__main__":
    # Path to your pre-trained model
    model_path = 'movie_recommendation_model(1).pth'
    
    # Run evaluation
    evaluate_recommendation_system(model_path)