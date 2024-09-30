import numpy as np
import pandas as pd

x = pd.read_csv('./../../GD1_ids.dat', delimiter='\t')
print(x)
lines = np.sort(x['id'].valuesG)
with open("./../../GD1_ids_list.txt", "w") as file:
    for line in lines:
        file.write(str(line))
        file.write('\n')

file.close()
