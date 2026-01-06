import pandas as pd
import matplotlib.pyplot as plt
import os


# --- Filenames ---
# Inputs: The final computed results from the analysis pipeline
INPUT_FILENAME = 'sentiment_analysis_results.csv'
# Output: The path for saving the final visualization image
OUTPUT_CHART_NAME = 'final_comparison_analysis.png'

# Choosing colours with clean contrast.
SEGMENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def create_segmentation_chart(results_file: str, chart_file: str):
    """
    Loads the final sentiment results and generates a bar chart for presentation.

    The chart includes core conclusions:
    1. Mean sentiment across segments.
    2. The correlation between player value (segment) and review effort (length).

    Args:
        results_file (str): Path to the final CSV output from the analysis pipeline.
        chart_file (str): Path where the final PNG visualization will be saved.
    """
    print("--- Step 1: Loading and Preparing Data for Visualization ---")
    
    # 1. Load the final results DataFrame
    df = pd.read_csv(results_file)

    # 2. Calculate the mean metrics by segment
    df_comparison = df.groupby('segment', observed=True)[['compound', 'review_length']].mean().reset_index()

    # 3. Filter and clean segments for plotting
    # Keep only the active, "Value" segments (Q1, Q2, Q3, Q4, plus any merged groups)
    df_comparison_chart = df_comparison[
        df_comparison['segment'].str.contains('Value', na=False)
    ].copy()

    # Drop rows where the compound score is NaN 
    df_comparison_chart.dropna(subset=['compound'], inplace=True) 

    # Rename segments for clear axis labels in the final chart
    df_comparison_chart['segment'] = df_comparison_chart['segment'].replace({
        'Low_Value(Q1-Q2)': 'Q1-Q2 (Low-Mid)',
        'Mid_Value(Q3)': 'Q3 (Mid)',
        'Q4_High_Value': 'Q4 (High)',
        'Q3_Mid_Value': 'Q3 (Mid-Value)',
        'Q2_Mid_Value': 'Q2 (Mid-Value)',
        'Q1_Low_Value': 'Q1 (Low-Value)',
    })
    
    # Define the order for the segments 
    final_chart_order = [
        'Q1 (Low-Value)', 'Q2 (Mid-Value)', 'Q3 (Mid-Value)', 'Q1-Q2 (Low-Mid)', 'Q3 (Mid)', 'Q4 (High)'
    ]
    df_comparison_chart['segment'] = pd.Categorical(df_comparison_chart['segment'], 
                                                   categories=[c for c in final_chart_order if c in df_comparison_chart['segment'].unique()], 
                                                   ordered=True)
    df_comparison_chart.sort_values('segment', inplace=True)
    
    
    print("--- Step 2: Generating Bar Charts ---")

    # --- 4. Create the visualization (Two subplots) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) 
    fig.suptitle('Comparative Analysis: Sentiment and Effort Across Player Segments (2024)', fontsize=18, fontweight='bold')
    
    plot_data = df_comparison_chart['segment']
    
    # --- Plot 1: Mean VADER Compound Score (Sentiment) ---
    axes[0].bar(
        plot_data, 
        df_comparison_chart['compound'], 
        color=SEGMENT_COLORS[:len(plot_data)],
        edgecolor='black',
        alpha=0.8
    )
    axes[0].set_title('Sentiment Score by Player Segment', fontsize=14)
    axes[0].set_ylabel('Mean Compound Score (Range: -1.0 to +1.0)', fontsize=12)
    axes[0].set_xlabel('Player Segment (by Owned Games Count)', fontsize=12)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8, label='Neutral Sentiment (0.0)') # Highlight the zero line
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    
    # --- Plot 2: Mean Review Length (Effort/Bias Check) ---
    axes[1].bar(
        plot_data, 
        df_comparison_chart['review_length'], 
        color=SEGMENT_COLORS[:len(plot_data)],
        edgecolor='black',
        alpha=0.8
    )
    axes[1].set_title('Review Length (Effort) Bias Check', fontsize=14)
    axes[1].set_ylabel('Average Word Count', fontsize=12)
    axes[1].set_xlabel('Player Segment (by Owned Games Count)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    
    # Final Presentation Polish
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for labels
    
    # Save the final chart
    plt.savefig(chart_file)
    print(f"\nVisualization successfully saved to {chart_file}")

if __name__ == '__main__':
    # Resolve Path: script -> scripts folder -> project root -> outputs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'outputs')

    # Define full file paths
    results_path = os.path.join(output_dir, INPUT_FILENAME)
    chart_save_path = os.path.join(output_dir, OUTPUT_CHART_NAME)

    # Run the visualization function
    create_segmentation_chart(results_path, chart_save_path)