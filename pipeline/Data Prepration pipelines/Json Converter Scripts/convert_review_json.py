import csv
import json
from collections import defaultdict

# Input CSV file path
input_csv = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/iitp/SonyResearchMovieRecommendation/imdb_reviews.csv'
# Output JSON file path
output_json = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/iitp/SonyResearchMovieRecommendation/pipeline/output/json/grouped_reviews.json'

# Dictionary to store reviews grouped by imdb_id
reviews_by_movie = defaultdict(list)

# Read CSV and group reviews by imdb_id
with open(input_csv, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        imdb_id = row['imdb_id']
        
        # Strip and clean data (some fields might be empty strings)
        review = {
            'review_title': row.get('review_title', '').strip() or None,
            'review_rating': row.get('review_rating', '').strip() or None,
            'review_text': row.get('review_text', '').strip() or None,
            'reviewer_name': row.get('reviewer_name', '').strip() or None,
            'review_date': row.get('review_date', '').strip() or None
        }

        reviews_by_movie[imdb_id].append(review)

# Write to JSON
with open(output_json, mode='w', encoding='utf-8') as json_file:
    json.dump(reviews_by_movie, json_file, indent=4, ensure_ascii=False)

print(f"JSON saved to {output_json}")
