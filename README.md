# Gaming Profiles 2024: Player Segmentation & Sentiment Analysis

### Project overview
It is stereotypical to say that high-value gamers are more demanding and critical than other player groups, as such players are likely to have rich gaming experience, making their taste for games elevated and specialized. Therefore, this project aims to find out whether this idea is statistically true, by analyzing player data and game reviews on Steam, the biggest video games distributor online. 

**The core question:** Are high-value players (based on game purchase history) more likely to leave negative reviews compared to other player groups?

---

### Data Source:
This project uses the **Gaming Profiles 2025** dataset on Kaggle. The dataset contains 103k players’ purchase history and 1.2 million reviews of different games from October, 2010 to January, 2025. 

* **Source:** [Kaggle - Gaming Profiles 2025 (Steam)](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox/data?select=steam)
* **Required Files:** `reviews_*.csv` and `purchased_games.csv`.

---

Key Engineering Decisions
**A change of metric for player’s value assessment:** Initially, “total spending on game purchases” was the metric for segmentation of players. However, due to data sparsity issues in the raw price data, this approach was infeasible. Thus, “how many games a player bought” was used instead to determine a player’s commercial value and experience level. 

---

### Skills & Tools
* **Libraries:** Pandas, NumPy, Matplotlib
* **Functions:** VADER Sentiment Analysis, RegEx
* **Statistics:** Quartile segmentation and Pareto distribution

---

### Repository Structure and Execution
To recreate this analysis, the steps are:
1. **Data Setup:** create a folder at `/data/Gaming Profiles 2025/` and place the downloaded Kaggle CSV files inside (https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox/data?select=steam).
2. **Data Ingestion:** run `scripts/01_data_prep.py` that handles data ingestion, cleaning and library count. 
3. **Data Analysis:** run `scripts/02_analysis_pipeline.py` that handles player segmentation and VADER scoring.
4. **Run Visualization:** run `scripts/03_reporting_viz.py` that generates the final visualization chart.

---

### Key Findings
The analysis shows that **high value players are not much more demanding and critical** than other groups. High value players’ sentiment score was the second-highest in positivity across all quartiles. In addition, they tend to provide significantly **longer  reviews**, indicating that more effort is being made. This suggests that high value players tend to be “detailed critics” instead of “harsh complainers”.

---

### Future Improvements of the Project
Two major changes that are out of the projects’ current scope but would contribute to a more accurate and nuanced analysis are:
* **Replacing VADER with LLMs:** While VADER can be a useful tool for analyzing English text, it becomes less reliable for analyzing non-English reviews (such as Chinese or Japanese reviews). In addition, reviews can include gaming community slang as well as Internet memes that VADER fails to properly detect, while LLMs provide a better understanding that is highly accurate for analysis.
* **Multi-Cultural Metrics:** Incorporating data regarding country, language, or genres of game could lead to a more nuanced analysis regarding the differences in player attitudes across varied cultural groups. This would allow a more precise segmentation of players, and increase the practical value of the project, since the results would be better aligned with real-world commercial strategy-making process.

---

### AI Usage
AI served as an educational and debugging tool for this project, primarily helping with:
* **Debugging:** Resolving Pandas and Matplotlib errors and warnings.
* **Refactoring:** Ensuring the pipeline follows professional Python standards and modular design.
