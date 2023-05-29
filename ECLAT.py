import pandas as pd
from preprocessing import data
from pyECLAT import ECLAT

data = pd.DataFrame(data)

min_threshold = 2

# we want to set min support to 7
# but we have to express it as a percentage
min_support = 7/len(data)

# we have no limit on the size of association rules
# so we set it to the longest transaction
max_length = max([len(x) for x in data])

# create an instance of eclat
my_eclat = ECLAT(data=data, verbose=True)

# fit the algorithm
rule_indices, rule_supports = my_eclat.fit(min_support=min_support,
                                           min_combination=min_threshold,
                                           max_combination=max_length)
print(rule_supports)