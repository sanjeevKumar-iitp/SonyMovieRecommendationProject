import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['combined_text'] = df[['storyline', 'description', 'tagline']].fillna('').agg(' '.join, axis=1)
    print(f"✅ Loaded {len(df)} movies.")
    return df


def vectorize_text(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df['combined_text'])
    print(f"🔠 TF-IDF vector shape: {X.shape}")
    return X, vectorizer


def find_optimal_clusters(X, k_range=range(2, 15)):
    inertia = []
    silhouette_scores = []

    print("🔍 Finding optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"   k={k}: inertia={kmeans.inertia_:.2f}, silhouette={score:.3f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (SSE)')
    plt.title('🔍 Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('📊 Silhouette Score')

    plt.tight_layout()
    plt.show()

    return inertia, silhouette_scores, k_range


def find_elbow_point(inertia_values, k_range):
    deltas = []
    for i in range(1, len(inertia_values)):
        delta = inertia_values[i - 1] - inertia_values[i]
        deltas.append(delta)

    acceleration = [deltas[i - 1] - deltas[i] for i in range(1, len(deltas))]
    optimal_k = k_range[2 + acceleration.index(max(acceleration))]
    return optimal_k


def cluster_and_plot(df, X, optimal_k):
    print(f"\n📦 Clustering with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(6, 4))
    df['cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.xlabel("Cluster")
    plt.ylabel("Number of Movies")
    plt.title("🎬 Histogram of Movies by Cluster")
    plt.show()

    return df, kmeans


def print_clusters(df, top_n=5):
    print("\n📌 Movies in Each Cluster:")
    for cluster_id in sorted(df['cluster'].unique()):
        print(f"\nCluster {cluster_id}:")
        cluster_movies = df[df['cluster'] == cluster_id]['title'].head(top_n)
        for idx, title in enumerate(cluster_movies, 1):
            print(f"   {idx}. {title}")


def print_top_features(vectorizer, kmeans, feature_names, num_top_features=10):
    print("\n🔍 Top Words Per Cluster:")
    for i, center in enumerate(kmeans.cluster_centers_):
        top_words = [feature_names[j] for j in center.argsort()[-num_top_features:][::-1]]
        print(f"\nCluster {i} top words: {', '.join(top_words)}")


# ✅ NEW: Genre-based Clustering Function
def cluster_by_genres(df):
    print("\n🎭 Clustering based on Genres...")

    # Step 1: One-hot encode the pipe-separated genres
    df['genre_list'] = df['genres'].fillna('').apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre_list'])

    print(f"✅ Genre vector shape: {genre_matrix.shape} ({len(mlb.classes_)} unique genres)")

    # Step 2: Find optimal K using Elbow + Silhouette
    inertia = []
    silhouette_scores = []
    k_range = range(2, 15)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(genre_matrix)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(genre_matrix, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"   k={k}: inertia={kmeans.inertia_:.2f}, silhouette={score:.3f}")

    # Plot Elbow Method
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (SSE)')
    plt.title('🎯 Elbow Method (Genre)')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('📈 Silhouette Score (Genre)')

    plt.tight_layout()
    plt.show()

    # Step 3: Find optimal K
    deltas = [inertia[i - 1] - inertia[i] for i in range(1, len(inertia))]
    acceleration = [deltas[i - 1] - deltas[i] for i in range(1, len(deltas))]
    optimal_k = k_range[2 + acceleration.index(max(acceleration))]
    print(f"\n🎯 Optimal K (Genre): {optimal_k}")

    # Step 4: Final Clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['genre_cluster'] = final_kmeans.fit_predict(genre_matrix)

    # Step 5: Print clusters and dominant genres
    print("\n📌 Genre-Based Clusters:")
    for cluster_id in sorted(df['genre_cluster'].unique()):
        cluster_indices = df['genre_cluster'] == cluster_id
        cluster_genres = genre_matrix[cluster_indices]
        mean_vector = cluster_genres.mean(axis=0)
        top_genres = [mlb.classes_[i] for i in mean_vector.argsort()[::-1][:3] if mean_vector[i] > 0]
        print(f"\nCluster {cluster_id} (Top Genres: {', '.join(top_genres)})")
        sample_titles = df[cluster_indices]['title'].head(5).tolist()
        for i, title in enumerate(sample_titles, 1):
            print(f"   {i}. {title}")

    return df

if __name__ == "__main__":
    csv_path = 'imdb_movie_metadata.csv'

    df = load_and_prepare_data(csv_path)
    X, vectorizer = vectorize_text(df)
    inertia, silhouette_scores, k_range = find_optimal_clusters(X)
    optimal_k = find_elbow_point(inertia, k_range)
    print(f"\n🎯 Optimal K using Elbow Method: {optimal_k}")
    clustered_df, kmeans_model = cluster_and_plot(df, X, optimal_k)
    print_clusters(clustered_df, top_n=5)
    feature_names = vectorizer.get_feature_names_out()
    print_top_features(vectorizer, kmeans_model, feature_names, num_top_features=10)
    clustered_df.to_csv("imdb_clustered_movies.csv", index=False)
    print("\n💾 Clustered data saved to 'imdb_clustered_movies.csv'")

    # ✅ Run Genre-based Clustering
    cluster_by_genres(df)
