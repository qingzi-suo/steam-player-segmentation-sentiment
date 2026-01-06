import pandas as pd
import os
import re
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# --- Filenames ---
INPUT_REVIEWS = 'clean_reviews_data.csv'
INPUT_LIBRARY = 'clean_library_data.csv'
OUTPUT_NAME = 'sentiment_analysis_results.csv'


def load_and_prepare_data(input_path):
    """
    Loads clean data files and carries out initial date and data quality filtering.

    Returns:
        tuple: (df_reviews_2024, df_library), the 2024 reveiws and the library counts.
    """
    print("--- Step 1: Data Loading and Preprocessing ---")
    
    rev_path = os.path.join(input_path, INPUT_REVIEWS)
    lib_path = os.path.join(input_path, INPUT_LIBRARY)

    df_reviews = pd.read_csv(rev_path)
    df_library = pd.read_csv(lib_path)
    
    # Convert 'posted' to datetime for correct filtering
    df_reviews['posted'] = pd.to_datetime(df_reviews['posted'])
    
    # Find 2024 reviews only
    df_reviews_2024 = df_reviews[df_reviews['posted'].dt.year == 2024].copy()
    print(f"1. Reviews filtered to 2024: {len(df_reviews_2024):,} rows.")
    
    # Drop reviews that are missing content/text
    initial_rows = len(df_reviews_2024)
    df_reviews_2024.dropna(subset=['review'], inplace=True)
    print(f"2. Missing reviews dropped: {initial_rows - len(df_reviews_2024):,} rows.")
    
    return df_reviews_2024, df_library


def segment_and_filter(df_reviews_2024, df_library):
    """
    1. Rank games based on their review count. 
    2. Segement player based on their library count. 
    3. Create subset for final comparison.

    Segmentation is based ONLY on plyaers active in 2024.

    Args:
        df_reviews_2024 (pd.DataFrame): reviews posted in 2024.
        df_library (pd.DataFrame): all players' library counts.

    Returns:
        pd.DataFrame: the final subset for sentiment analysis.

    """
    print("\n--- Step 2: Segmentation and Filtering ---")
    
    # Find Most Reviewed Games based on Pareto Principle (80/20 Rule)
    review_counts = df_reviews_2024['gameid'].value_counts()
    cum_percent = review_counts.cumsum() / review_counts.sum()
    
    # Select games that countribute to the top 80% of review volume
    pareto_threshold = cum_percent[cum_percent <= 0.8].index
    most_reviewed_gameids = pareto_threshold.tolist()
    
    print(f"3. Found {len(most_reviewed_gameids):,} games that account for 80% of 2024 review volume.")
    
    
    # Player segmentation based on 2024 activity
    target_playerids = df_reviews_2024['playerid'].unique()
    df_segmentation_data = df_library[df_library['playerid'].isin(target_playerids)].copy()
    
    print(f"4. Segmentation pool created: {len(df_segmentation_data):,} players active in 2024.")
    
    # Segment Players by Owned Game Count (Quartiles) -
    library_counts = df_segmentation_data['owned_games_count']
    q_bins = library_counts.quantile([0.25, 0.50, 0.75]).tolist()
    

    bins = [library_counts.min() - 1] + q_bins + [library_counts.max() + 1]
    unique_bins = sorted(list(set(bins)))
    num_bins = len(unique_bins) - 1 
    
    # Define types of labels based on the actual number of segments that can be created
    if num_bins == 4:
         final_labels = ['Q1_Low_Value', 'Q2_Mid_Value', 'Q3_Mid_Value', 'Q4_High_Value']
    elif num_bins == 3:
         # This case occurs when Q1 and Q2 quantiles are identical (both 0)
         final_labels = ['Low_Value(Q1-Q2)', 'Mid_Value(Q3)', 'High_Value(Q4)']
    else: # Fallback for extreme cases
         final_labels = [f'Segment_{i+1}' for i in range(num_bins)]

    # Apply pd.cut with unique bins
    df_segmentation_data['segment'] = pd.cut(
        library_counts, 
        bins=unique_bins,
        labels=final_labels[:num_bins], 
        include_lowest=True, 
        right=True
    )
    
    # Merge the segment label back onto the full library table for later merging with reviews
    df_library_segmented = df_library.merge(
        df_segmentation_data[['playerid', 'segment']],
        on='playerid',
        how='left'
    )

    # A engineering fix has been made: must add the category before using fillna on a categorical Dtype
    df_library_segmented['segment'] = df_library_segmented['segment'].cat.add_categories('Inactive_2024')
    df_library_segmented['segment'] = df_library_segmented['segment'].fillna('Inactive_2024')
    
    # Identify the high-value players (Q4) for general metric reporting
    high_value_players = df_library_segmented[
        df_library_segmented['segment'].str.contains('High_Value|Q4_High_Value', na=False)
    ]['playerid'].unique()
    
    print(f"5. Players segmented based on 2024 active pool. High-Value player count: {len(high_value_players):,}")


    # --- Final subset for sentiment analysis and comparison---
 
    # Filter to include all reviews on the most-reviews games 
    df_subset = df_reviews_2024[
        df_reviews_2024['gameid'].isin(most_reviewed_gameids)
    ].copy()

    # Merge the segment label (Q1, Q2, Q3, Q4) onto the subset
    segment_map = df_library_segmented.set_index('playerid')['segment']
    df_subset['segment'] = df_subset['playerid'].map(segment_map)
    
    # prevent categorical NaN issue again before filtering
    df_subset['segment'] = df_subset['segment'].cat.add_categories('Missing_Segment') # Add category first
    df_subset['segment'] = df_subset['segment'].fillna('Missing_Segment')

    # keep only segments containing 'Value' (Q1, Q2, Q3, Q4, or merged groups)
    segments_to_keep = [s for s in df_subset['segment'].unique() if 'Value' in str(s)]
    df_subset = df_subset[df_subset['segment'].isin(segments_to_keep)].copy()
    print(f"6. Final analysis subset created: {len(df_subset):,} reviews.")
    
    return df_subset



def clean_and_analyze_sentiment(df_subset):
    """
    1. RegEx cleaning
    2. VADER analysis
    3. Comparative bias check

    Args: 
        df_subset (pd.DataFrame): the filtered review subset.
    
    Returns:
        pd.DataFrame: the final DataFrame including VADER scores and review length.
    """
    print("\n--- Step 3: Sentiment Analysis and Bias Check ---")
    
    # --- RegEx cleaning ---
    # Define a RegEx pattern to clean noise (Steam tags and URLs)
    tag_url_pattern = r'\[.*?\]|https?://\S+|www\.\S+'
    
    def clean_review_text(text):
        if pd.isna(text):
            return ""
        # Remove tags/URLs
        text = re.sub(tag_url_pattern, '', text, flags=re.IGNORECASE)
        return text

    df_subset['review_clean'] = df_subset['review'].apply(clean_review_text)
    print("7. RegEx cleaning applied to review text.")


    # --- VADER Sentiment Analysis ---
    sia = SentimentIntensityAnalyzer()
    # Apply VADER and store all polarity scores (neg, neu, pos, compound)
    df_subset['vader_dictionary'] = df_subset['review_clean'].apply(sia.polarity_scores)
    # Expand the dictionary into separate columns for easy analysis
    df_subset[['neg', 'neu', 'pos', 'compound']] = df_subset['vader_dictionary'].apply(pd.Series)
    df_subset.drop(columns=['vader_dictionary'], inplace=True)
    print("8. VADER scores (neg, neu, pos, compound) calculated.")


    # --- Comparative Analysis and Bias Check ---
    # Calculate Review Length (Word Count) for bias analysis
    df_subset['review_length'] = df_subset['review_clean'].apply(lambda x: len(x.split()))
    
    # Final comparative results table for Q1-Q4 segments
    df_comparison = df_subset.groupby('segment', observed=True)[['compound', 'review_length']].mean().reset_index()
    
    print("\n--- Summary of Sentiment and Length by Player Segment ---")
    print(df_comparison.to_markdown(index=False, floatfmt=".3f"))
    
    return df_subset


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, 'outputs')
    output_dir = os.path.join(project_root, 'outputs')

    # Check output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run analysis pipeline
    df_2024, df_lib = load_and_prepare_data(input_dir)
    df_filtered = segment_and_filter(df_2024, df_lib)
    df_results = clean_and_analyze_sentiment(df_filtered)
    
    # Save Results
    final_output_path = os.path.join(output_dir, OUTPUT_NAME)
    df_results.to_csv(final_output_path, index=False)
    print(f"\n Analysis complete. Results saved to: {final_output_path}")