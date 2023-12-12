import os
import glob
import pandas as pd

path = "/volumes1/xai_cl/resnet_l64/er"

extension = 'csv'
all_filenames = [i for i in glob.glob(path + '/*/{}.{}'.format("descriptions", extension))]
combined_csv = []

combined_csv = [pd.read_csv(f, sep=',') for f in all_filenames ]
combined_csv = pd.concat(combined_csv,  axis=0) #ignore_index=True)
#export to csv
combined_csv.to_csv(os.path.join(path,"combined_alllayers.csv"), index=False, encoding='utf-8-sig')
