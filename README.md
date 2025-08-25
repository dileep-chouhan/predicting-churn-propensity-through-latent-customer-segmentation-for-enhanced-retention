# Predicting Churn Propensity Through Latent Customer Segmentation for Enhanced Retention

## Overview

This project aims to predict customer churn propensity for a retail business by identifying latent customer segments exhibiting distinct behavioral patterns.  The analysis leverages unsupervised machine learning techniques to segment customers based on their transactional and demographic data.  This allows for the identification of high-risk segments most likely to churn, enabling proactive retention strategies and ultimately minimizing revenue loss.  The project includes data preprocessing, model training, segment profiling, and visualization of key findings.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required Python libraries listed above using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print key analysis results to the console, including details about the identified customer segments, their churn probabilities, and relevant descriptive statistics.  Additionally, the script will generate several visualization files (e.g., `customer_segment_distribution.png`, `churn_probability_by_segment.png`) in the `output` directory, illustrating the identified segments and their churn probabilities.  These visualizations provide a clear visual representation of the findings and facilitate insightful interpretation of the results.  The specific output files generated may vary slightly depending on the data and the model's performance.