import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import os
import re

# Load JSON data
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Extract valid reviews
def get_movie_reviews(reviews, max_reviews=10):
    return [rev['review_text'] for rev in reviews[:max_reviews] if rev.get('review_text')]

# Build structured prompt
def build_prompt(movie_title, year_of_release, reviews):
    prompt = f"""<|begin_of_sentence|>Suppose you are an expert at summarizing user feedback about movies. You will generate two short, focused summaries based on the following user reviews of the movie "{movie_title}" ({year_of_release}).

Your task is to extract:
1. A **Positive Summary** highlighting what users liked most.
2. A **Negative Summary** outlining what users disliked or found disappointing.

Read all reviews carefully. For each summary:
- Keep it under 100 words.
- Make sure it's clear, factual, and reflects the general sentiment in the reviews.
- Do not include opinions outside the given reviews.
- Write each summary independently as if you’re doing them separately.

Now read these reviews:

"""
    for i, r in enumerate(reviews, start=1):
        prompt += f"{i}. {r}\n"

    prompt += """
Positive Summary:
"""
    return prompt

# Build negative prompt after positive one
def build_negative_prompt(movie_title, year_of_release, reviews, positive_summary):
    prompt = f"""<|begin_of_sentence|>Suppose you are an expert at summarizing user feedback about movies. You will now generate a concise **negative summary** for the movie "{movie_title}" ({year_of_release}), focusing only on criticisms and drawbacks mentioned in the reviews.

Ignore any positive feedback already summarized below:
"{positive_summary}"

Now read the same reviews again and write a short, focused summary of what users disliked or found disappointing. Limit it to 100 words.

Negative Summary:
"""
    return prompt

# Parse raw LLM response
def parse_summary(output):
    # Remove leading/trailing whitespace and extra lines
    summary = re.sub(r'\s+', ' ', output.strip())
    return summary[:500]  # Limit to 500 chars (~100 words)

# Run generation
def generate_summary(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_summary(response)
    except Exception as e:
        print("Error during generation:", str(e))
        return ""

# Main pipeline function
def run_pipeline(
    json_path,
    output_file="movie_summaries.json",
    MAX_MOVIES=100,
    START_OFFSET=0,
    MAX_REVIEWS_PER_MOVIE=5,
    MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
):
    data = load_data(json_path)

    imdb_ids = list(data.keys())[START_OFFSET:]
    total_movies = min(MAX_MOVIES, len(imdb_ids))

    device = "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"Processing up to {total_movies} movies starting from index {START_OFFSET}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    results = []

    for idx, imdb_id in tqdm(enumerate(imdb_ids), desc="Processing Movies", total=total_movies):
        if idx >= total_movies:
            break

        reviews = data[imdb_id]
        selected_reviews = get_movie_reviews(reviews, max_reviews=MAX_REVIEWS_PER_MOVIE)

        if not selected_reviews:
            print(f"⚠️ No valid reviews found for movie {imdb_id}")
            continue

        # Extract title and optional year
        movie_title = imdb_id
        year_of_release = ""
        if 'review_title' in reviews[0]:
            full_title = reviews[0]['review_title']
            if "(" in full_title and ")" in full_title:
                parts = full_title.split("(")
                movie_title = " ".join(parts[:-1]).strip()
                year_of_release = parts[-1].replace(")", "").strip()

        # Generate positive summary
        prompt_pos = build_prompt(movie_title, year_of_release, selected_reviews)
        pos_summary = generate_summary(model, tokenizer, prompt_pos, device)

        # Generate negative summary
        prompt_neg = build_negative_prompt(movie_title, year_of_release, selected_reviews, pos_summary)
        neg_summary = generate_summary(model, tokenizer, prompt_neg, device)

        results.append({
            "imdb_id": imdb_id,
            "title": movie_title,
            "year": year_of_release,
            "positive_summary": pos_summary,
            "negative_summary": neg_summary
        })

        # Memory cleanup
        del reviews, selected_reviews, prompt_pos, prompt_neg
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    csv_file = output_file.replace(".json", ".csv")
    df.to_csv(csv_file, index=False)

    print(f"✅ Summaries saved to {output_file} and {csv_file}")

if __name__ == "__main__":
    run_pipeline(
        json_path="/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/pipeline/Data Prepration pipelines/output/json/grouped_reviews.json",
        output_file="summaries_part1.json",
        MAX_MOVIES=2,               # Change this to 100, 1000, etc.
        START_OFFSET=0,              # Resume from where you left off
        MAX_REVIEWS_PER_MOVIE=5   # Use up to N reviews per movie
    )