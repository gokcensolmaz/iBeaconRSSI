import pyfpgrowth
from preprocessing import data

# Convert the dataset to a list of transactions
transactions = data.values.tolist()

# Perform FP-Growth frequent pattern mining
patterns = pyfpgrowth.find_frequent_patterns(transactions, 3)

print("FP-Growth with min threshold = 3; where is the frequency greater than 1335:")
count = 0
# Print the frequent patterns
for pattern, support in patterns.items():
    count += 1
    if support > 1335:
        print(f"Pattern: {pattern}, Support: {support}")
print("\n Total number of result: ", count)
