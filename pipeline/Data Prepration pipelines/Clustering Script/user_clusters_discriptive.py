# ----------------------------
# full_user_clustering_pipeline_with_kmeans_elbow_method_final_with_avg.py
# Batch User-Based Clustering with Hybrid Features + KMeans + Elbow Method + Cluster Labeling
# Now uses GPU (if available) for SentenceTransformer
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import os
import csv
import warnings
from tqdm import tqdm
import torch  # ✅ Added for GPU detection

warnings.filterwarnings("ignore")

# Set working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "..", "dataset")
IMDB_CSV = os.path.join(PROJECT_ROOT, "imdb_movie_metadata.csv")

# Output directories and files
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_KMEANS_REPORT_CSV = os.path.join(OUTPUT_DIR, "user_kmeans_report.csv")
USER_MOVIE_CLUSTERS_CSV = os.path.join(OUTPUT_DIR, "user_movie_clusters.csv")
CLUSTER_INTERPRETATIONS_CSV = os.path.join(OUTPUT_DIR, "user_cluster_interpretations.csv")
CLUSTER_LABELS_CSV = os.path.join(OUTPUT_DIR, "user_cluster_labels.csv")

# 🔧 CONFIGURATION
MAX_USERS_TO_PROCESS = 10

# ----------------------------
# STEP 1: Load MovieLens Data
# ----------------------------
def load_movielens_data():
    print("[INFO] Loading MovieLens data...")
    movies = pd.read_csv(os.path.join(DATASET_DIR, "movies.dat"),
                         sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'],
                         encoding='latin-1')
    ratings = pd.read_csv(os.path.join(DATASET_DIR, "ratings.dat"),
                          sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    users = pd.read_csv(os.path.join(DATASET_DIR, "users.dat"),
                        sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode'],
                        encoding='latin-1')
    return movies, ratings, users

# ----------------------------
# STEP 2: Load IMDb Metadata
# ----------------------------
def load_imdb_metadata():
    df = pd.read_csv(IMDB_CSV)
    print(f"[SUCCESS] Loaded IMDb metadata for {len(df)} movies.")
    return df

# ----------------------------
# STEP 3: Preprocessing Helper
# ----------------------------
def preprocess_text(text):
    if not isinstance(text, str) or pd.isna(text): return "unknown"
    return text.strip().lower()

# ----------------------------
# STEP 4: Feature Encoders
# ----------------------------

# ✅ Check and use GPU for sentence transformer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device for SentenceTransformer: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def encode_genres_tags(data_series):
    sentences = []
    for x in data_series:
        if isinstance(x, str):
            tags = [t.strip() for t in x.lower().split('|') if t.strip()]
            sentences.append(tags)
        else:
            sentences.append([])
    model = Word2Vec(sentences=[s for s in sentences if s], vector_size=100,
                     window=5, min_count=1, workers=4)

    vectors = []
    for tags in sentences:
        if tags:
            vecs = [model.wv[tag] for tag in tags if tag in model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(100))
        else:
            vectors.append(np.zeros(100))
    return np.array(vectors)


def encode_sentences(sentence_list):
    cleaned = [str(s) if isinstance(s, str) and s.strip() != '' else "unknown" for s in sentence_list]
    return _model.encode(cleaned)


def tfidf_features(text_list):
    cleaned = [str(t) if isinstance(t, str) and t.strip() != '' else "unknown" for t in text_list]
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(cleaned).toarray()

# ----------------------------
# STEP 5: Find Optimal K Using Elbow Method
# ----------------------------
def find_optimal_k_elbow(X, max_k=10):
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("No samples provided for clustering.")

    max_k = min(max_k, n_samples)
    if max_k < 2:
        max_k = 1

    k_range = range(1, max_k + 1)
    inertias = []

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        except Exception as e:
            inertias.append(float('inf'))

    if len(inertias) == 0:
        return {"optimal_k": 1, "inertias": []}

    try:
        deltas = np.diff(inertias)
        acc = np.diff(deltas)
        optimal_k = k_range[np.argmin(acc) + 1]
    except ValueError:
        optimal_k = 1

    return {
        "optimal_k": optimal_k,
        "inertias": inertias
    }

# ----------------------------
# STEP 6: Analyze One User
# ----------------------------
def analyze_user(user_id, movies_df, ratings_df, users_df, imdb_df):
    try:
        print(f"\n[DEBUG] Processing User ID: {user_id}")

        # Step 1: Get watched movies
        watched = ratings_df[ratings_df['UserID'] == user_id]
        merged = pd.merge(watched, movies_df, on='MovieID', how='inner')
        merged['CleanTitle'] = merged['Title'].str.extract(r'^(.*?)(?: $|\()')[0].str.strip()
        matched = pd.merge(merged[['CleanTitle']], imdb_df,
                           left_on='CleanTitle', right_on='title', how='inner')

        if len(matched) < 1:
            print(f"[WARNING] Skipping user {user_id}: No IMDb matches found.")
            return None, None, None, None

        # Clean text fields
        cols_to_preprocess = ['genres', 'storyline', 'description', 'directors', 'cast']
        for col in cols_to_preprocess:
            matched[col] = matched[col].apply(preprocess_text)

        # Encode features
        genre_vectors = encode_genres_tags(matched['genres'])
        story_vectors = encode_sentences(matched['storyline'])
        desc_vectors = encode_sentences(matched['description'])
        director_tfidf = tfidf_features(matched['directors'])
        cast_tfidf = tfidf_features(matched['cast'])

        # Combine all features
        combined_features = np.hstack([
            genre_vectors,
            story_vectors,
            desc_vectors,
            director_tfidf,
            cast_tfidf
        ])

        # Analysis A - All Features
        print("\n📌 ANALYSIS A: Clustering using ALL FEATURES")
        result_a = find_optimal_k_elbow(combined_features)
        print(f"[RESULT] Optimal K (All Features): {result_a['optimal_k']}")

        # Analysis B - Only Genre
        print("\n📌 ANALYSIS B: Clustering using ONLY GENRE")
        result_b = find_optimal_k_elbow(genre_vectors)
        print(f"[RESULT] Optimal K (Genre Only): {result_b['optimal_k']}")

        # Analysis C - No Genre
        no_genre_features = np.hstack([
            story_vectors,
            desc_vectors,
            director_tfidf,
            cast_tfidf
        ])
        print("\n📌 ANALYSIS C: Clustering WITHOUT GENRE")
        result_c = find_optimal_k_elbow(no_genre_features)
        print(f"[RESULT] Optimal K (No Genre): {result_c['optimal_k']}")

        # Use All Features for final clustering
        k_final = result_a["optimal_k"]
        if k_final > len(matched):
            k_final = len(matched)

        kmeans = KMeans(n_clusters=k_final, random_state=42)
        labels = kmeans.fit_predict(combined_features)

        # Build output rows
        titles = matched['title'].tolist()
        movie_cluster_list = []
        for idx, cluster_id in enumerate(labels):
            movie_cluster_list.append({
                "user_id": user_id,
                "movie_title": titles[idx],
                "cluster_id": int(cluster_id)
            })

        # Interpret clusters
        interpretations = interpret_clusters(matched, labels, k_final, user_id)

        # Label clusters
        labels_info = label_clusters(interpretations)

        # Get user demographic info
        user_info = users_df[users_df['UserID'] == user_id].iloc[0]

        watched_count = len(matched)

        user_report = {
            "user_id": user_id,
            "gender": user_info['Gender'],
            "age": user_info['Age'],
            "occupation": user_info['Occupation'],
            "watched_count": watched_count,
            "a_optimal_k": result_a["optimal_k"],
            "b_optimal_k": result_b["optimal_k"],
            "c_optimal_k": result_c["optimal_k"],
            "avg_count_all": round(watched_count / result_a["optimal_k"], 2),
            "avg_count_genre_only": round(watched_count / result_b["optimal_k"], 2),
            "avg_count_no_genre": round(watched_count / result_c["optimal_k"], 2)
        }

        return user_report, movie_cluster_list, interpretations, labels_info

    except Exception as e:
        print(f"[ERROR] Failed to process user {user_id}: {e}")
        return None, None, None, None


# ----------------------------
# STEP 7: Interpret Clusters
# ----------------------------
def interpret_clusters(df, labels, k, user_id):
    interpretations = []
    movie_titles = df['title'].tolist()

    df['cluster'] = labels

    for cluster_id in range(k):
        cluster_movies = df[df['cluster'] == cluster_id]
        titles = cluster_movies['title'].tolist()
        genres = "|".join(cluster_movies['genres']).split("|")
        directors = "|".join(cluster_movies['directors']).split("|")
        cast = "|".join(cluster_movies['cast']).split("|")

        interpretations.append({
            "user_id": user_id,
            "cluster_id": cluster_id,
            "sample_movies": "; ".join(titles[:3]),
            "dominant_genres": ", ".join(pd.Series(genres).value_counts().head(2).index.tolist()),
            "dominant_directors": ", ".join(pd.Series(directors).value_counts().head(2).index.tolist()),
            "dominant_cast": ", ".join(pd.Series(cast).value_counts().head(2).index.tolist())
        })

    return interpretations


# ----------------------------
# STEP 8: Label Clusters
# ----------------------------
def label_clusters(interpretations):
    labels = []
    for item in interpretations:
        genre_label = item['dominant_genres'].strip()
        director_label = item['dominant_directors'].strip()
        cast_label = item['dominant_cast'].strip()

        label_parts = []

        if genre_label:
            label_parts.append(genre_label.title())

        if director_label and director_label not in genre_label:
            label_parts.append(f"{director_label.title()} Films")

        elif cast_label and cast_label not in genre_label:
            label_parts.append(f"{cast_label.split(',')[0]} Movies")

        final_label = " / ".join(label_parts[:2]) if label_parts else "Unknown Theme"

        labels.append({
            "user_id": item['user_id'],
            "cluster_id": item['cluster_id'],
            "label": final_label
        })

    return labels


# ----------------------------
# STEP 9: Main Batch Runner
# ----------------------------
def run_full_analysis():
    print("🚀 Starting full batch KMeans analysis...")

    movies_df, ratings_df, users_df = load_movielens_data()
    imdb_df = load_imdb_metadata()

    # Ensure we pick users with IDs from 0 to MAX_USERS_TO_PROCESS - 1
    unique_users = np.arange(MAX_USERS_TO_PROCESS)
    print(f"🔢 Total users to process: {len(unique_users)}")

    # Prepare CSV writers
    report_fields = ["user_id", "gender", "age", "occupation", "watched_count",
                     "a_optimal_k", "b_optimal_k", "c_optimal_k",
                     "avg_count_all", "avg_count_genre_only", "avg_count_no_genre"]

    cluster_fields = ["user_id", "movie_title", "cluster_id"]
    interpretation_fields = ["user_id", "cluster_id", "sample_movies",
                             "dominant_genres", "dominant_directors", "dominant_cast"]
    label_fields = ["user_id", "cluster_id", "label"]

    def init_csv(file_path, fieldnames):
        if not os.path.isfile(file_path):
            with open(file_path, mode="w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    init_csv(USER_KMEANS_REPORT_CSV, report_fields)
    init_csv(USER_MOVIE_CLUSTERS_CSV, cluster_fields)
    init_csv(CLUSTER_INTERPRETATIONS_CSV, interpretation_fields)
    init_csv(CLUSTER_LABELS_CSV, label_fields)

    for user_id in tqdm(unique_users, desc=f"🧠 Processing First {MAX_USERS_TO_PROCESS} Users"):
        user_report, movie_clusters, interpretations, labels = analyze_user(
            user_id, movies_df, ratings_df, users_df, imdb_df
        )

        if user_report:
            with open(USER_KMEANS_REPORT_CSV, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=report_fields)
                writer.writerow(user_report)

        if movie_clusters:
            with open(USER_MOVIE_CLUSTERS_CSV, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cluster_fields)
                writer.writerows(movie_clusters)

        if interpretations:
            with open(CLUSTER_INTERPRETATIONS_CSV, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=interpretation_fields)
                writer.writerows(interpretations)

        if labels:
            with open(CLUSTER_LABELS_CSV, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=label_fields)
                writer.writerows(labels)

    print(f"📂 All results saved to: {OUTPUT_DIR}")


# ----------------------------
# STEP 10: Run It!
# ----------------------------
if __name__ == "__main__":
    run_full_analysis()