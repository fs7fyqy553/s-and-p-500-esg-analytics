from csv import DictReader
import numpy as np

# Read the data from the CSV file
with open('./trimmedData.csv') as f:
    # Skipping the header
    next(f)

    fieldnames = ['Symbol', 'Sector', 'Industry', 'Full Time Employees', 'Total ESG Risk score', 'Environment Risk Score', 'Governance Risk Score', 'Social Risk Score', 'Controversy Level', 'Controversy Score', 'ESG Risk Percentile', 'ESG Risk Level']
    data = [r for r in DictReader(f, fieldnames)]

# Extract and process the data
sectors = []
score_lists = {
    'Environment Risk Score': [],
    'Social Risk Score': [],
    'Governance Risk Score': []
}

for company in data:
    sectors.append(company['Sector'])
    for score_label in score_lists.keys():
        score_lists[score_label].append(float(company[score_label]))

# Calculate Basic Statistical Summary
summary_stats = {}
for score_label, scores in score_lists.items():
    scores = np.array(scores)  # Convert to NumPy array for easier computation
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    summary_stats[score_label] = {
        'Minimum': np.min(scores),
        'Q1': q1,
        'Median (Q2)': np.percentile(scores, 50),
        'Q3': q3,
        'Maximum': np.max(scores),
        'Q3 + 1.5 * IQR': q3 + 1.5 * iqr  # Upper threshold for outliers
    }

# Print the results
for score_label, stats in summary_stats.items():
    print(f"{score_label} Summary:")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")
    print()
