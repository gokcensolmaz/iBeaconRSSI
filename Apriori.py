from preprocessing import data
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# Convert the dataset into a binary format suitable for Apriori
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# Apply one-hot encoding to the dataset
data_encoded = data.applymap(encode_units)

frequent_patterns = apriori(data_encoded, min_support=0.05, use_colnames=True)

association_rules = association_rules(frequent_patterns, metric="confidence", min_threshold=0.3)

# Print the frequent patterns and association rules
print("Frequent Patterns with min support = 0.05:")
print(frequent_patterns)
print("\nAssociation Rules with min threshold = 0.3:")
print(association_rules)
