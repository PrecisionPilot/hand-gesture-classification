import pandas as pd
import seaborn as sns
import numpy as np

# process concatenated data to pass into seaborn pairplot
column_names = list
graph_combined_X = np.delete(combined_X, np.s_[-2:], axis=0) # remove last two rows
graph_combined_X = pd.DataFrame(graph_combined_X)
graph_combined_X['labels'] = combined_y
sns.pairplot(graph_combined_X, vars=['labels'], hue="labels")