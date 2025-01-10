from csv import DictReader
import matplotlib.pyplot as plt

with open('./trimmedData.csv') as f:
    # Skipping the header
    next(f)

    fieldnames = ['Symbol', 'Sector', 'Industry', 'Full Time Employees', 'Total ESG Risk score', 'Environment Risk Score', 'Governance Risk Score', 'Social Risk Score', 'Controversy Level', 'Controversy Score', 'ESG Risk Percentile', 'ESG Risk Level']
    data = [r for r in DictReader(f, fieldnames)]

# For the below lists, a common index is for a common company
sectors = []
score_lists = {
    'Environment Risk Score': [],
    'Social Risk Score': [],
    'Governance Risk Score': []
}

# Populating the above lists
for company in data:
    sectors.append(company['Sector'])
    for score_label in score_lists.keys():
        score_lists[score_label].append(float(company[score_label]))

for score_label in score_lists:
    scores = score_lists[score_label]
    data = {}

    # Preparing data to be plotted
    for sector, score in zip(sectors, scores):
        data.setdefault(sector, []).append(score)
    grouped_data = [data[key] for key in sorted(data.keys())]
    labels = sorted(data.keys())

    # Creating the boxplot
    plt.boxplot(grouped_data, labels=labels)
    plt.title(f'{score_label} Box Plots For Each Sector')

    # Boxplot formatting
    formatted_labels = ['\n'.join(label.split()) for label in labels]
    plt.xticks(ticks=range(1, len(formatted_labels) + 1), labels=formatted_labels, fontsize=7)
    plt.xlabel('Sector')
    plt.ylabel(score_label)
    
    plt.show()
