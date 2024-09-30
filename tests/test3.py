import numpy as np
import pandas as pd
import torch_config
import torch_dataloader
import os


opts = torch_config.config_ger_1000
data_dir = opts['data_path_bulk']
data_dict = {}
true_params = np.array([], dtype=np.float32).reshape(0,5)
output_dir_key = data_dir + 'map_files/'
all_params = pd.read_csv(opts['metadata_directory'])
if not os.path.exists(output_dir_key):
    os.makedirs(output_dir_key)
for i in range(11):
    data_dict_tmp, true_params_tmp = torch_dataloader.datareader(i, opts)
    for key in data_dict_tmp.keys():
        data = data_dict_tmp[key]
        data = np.log10(data * all_params.const[all_params.ID == key].values[0] + 0.001)
        data = torch_dataloader.NormalizeData(data, 6, -3).astype(np.float32)

        np.save(output_dir_key + 'map_'+str(key), data)

