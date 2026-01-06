import pandas as pd
import os
import glob
import re

OUTPUT_REVIEWS_FILE = 'clean_reviews_data.csv'
OUTPUT_LIBRARY_FILE = 'clean_library_data.csv'

def load_and_clean_data(data_path):
    """
    Loads raw review and player library data. A key change was made here during the project.
    Instead of using unreliable price data (high sparsity) to segment players,
    now 'owned_games_count' is used for player segmentation.

    Args:
        data_path: The directory where the raw CSV files are stored.

    Returns: 
        tuple: a tuple that contains the cleaned review dataframe and the player's library counts.
    """
    print("--- Step 1: Data Ingestion and Preparation ---")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found at: {data_path}")

    # --- Load Reviews Data ---
    review_files = glob.glob(os.path.join(data_path, "reviews*.csv"))
    if not review_files:
        print(f"Warning: no review files found in {data_path}")
    

    df_reviews_list = [
        pd.read_csv(f, usecols=['reviewid', 'playerid', 'gameid', 'review', 'posted']) 
        for f in review_files
    ]
    df_reviews = pd.concat(df_reviews_list, ignore_index=True)
    print(f"1. Raw reviews data loaded: {len(df_reviews):,} rows.")

    # --- Load Purchased Games (Library) ---
    df_library = pd.read_csv(os.path.join(data_path, "purchased_games.csv"), usecols=['playerid', 'library'])
    print(f"2. Raw player library data loaded: {len(df_library):,} rows.")

    # --- Count Owned Games in Player's Library ---
    
    def count_games(library_str):
        """
        Uses RegEx to safely count the number of game IDs. Returns 0 if data is missing.
        """

        if pd.isna(library_str):
            return 0
        # The library content is a string of game ids; re.findall extracts all numeric groups.
        return len(re.findall(r'\d+', library_str))


    # Apply the counting logic to create the primary segmentation feature.
    df_library['owned_games_count'] = df_library['library'].apply(count_games)
    print("3. 'owned_games_count' calculated for player segmentation.")
    
    # Drop the raw 'library' column to save memory space.
    df_library.drop(columns=['library'], inplace=True)
    
    return df_reviews, df_library


def save_cleaned_data(df_reviews, df_library, output_path):
    """
    Saves the two processed DataFrames to separate CSV files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")
    print("\n--- Step 2: Saving Clean Outputs ---")
    
    # Save the review data, later for filtering and sentiment analysis using VADER.
    full_reviews_path = os.path.join(output_path, OUTPUT_REVIEWS_FILE)
    full_library_path = os.path.join(output_path, OUTPUT_LIBRARY_FILE)
    df_reviews.to_csv(full_reviews_path , index=False)
    print(f"Reviews saved to {full_reviews_path} ({len(df_reviews):,} rows)")

    # Save the player ids and their owned library counts data.
    df_library.to_csv(full_library_path, index=False)
    print(f"Library counts saved to {full_library_path} ({len(df_library):,} rows)")


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    data_path = os.path.join(project_root, 'data', 'Gaming Profiles 2025')
    output_path = os.path.join(project_root, 'outputs')

    df_reviews, df_library = load_and_clean_data(data_path=data_path)
    save_cleaned_data(df_reviews, df_library, output_path=output_path)

    print("\n df_reviews and df_library are created.")