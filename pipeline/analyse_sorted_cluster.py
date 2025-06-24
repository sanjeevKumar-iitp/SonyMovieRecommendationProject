import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_sorted_clusters_from_csv(csv_path='output/user_sorted_clusters.csv', output_dir='output/analyse_chart'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path, parse_dates=['timestamp_readable'])
    
    # Convert timestamp_readable to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp_readable']):
        df['timestamp_readable'] = pd.to_datetime(df['timestamp_readable'])
    
    # Extract hour from timestamp_readable
    df['hour'] = df['timestamp_readable'].dt.hour
    
    # Group by each user
    for user_id, user_data in df.groupby('user_id'):
        plt.figure(figsize=(12, 8))

        # 1) Histogram: Number of movies per cluster for this user
        plt.subplot(2, 2, 1)
        cluster_counts = user_data['cluster_id'].value_counts().sort_index()
        cluster_counts.plot(kind='bar')
        plt.title(f'User {user_id} - Movies per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Movies')

        # 2) Histogram: Movie counts per hour of day
        plt.subplot(2, 2, 2)
        hour_counts = user_data['hour'].value_counts().sort_index()
        hour_counts.plot(kind='bar', color='orange')
        plt.title(f'User {user_id} - Movies Watched per Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Movies')

        # 3) Heatmap: Hour vs Cluster frequency matrix
        plt.subplot(2, 1, 2)
        heatmap_data = user_data.groupby(['hour', 'cluster_id']).size().unstack(fill_value=0)
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
        plt.title(f'User {user_id} - Watch Frequency Heatmap (Hour vs Cluster)')
        plt.xlabel('Cluster ID')
        plt.ylabel('Hour of Day')

        plt.tight_layout()

        # Save the figure instead of showing it
        save_path = os.path.join(output_dir, f'user_{user_id}_cluster_analysis.png')
        plt.savefig(save_path)
        plt.close()

        print(f'Saved analysis for user {user_id} at: {save_path}')

if __name__ == "__main__":
    analyse_sorted_clusters_from_csv()
