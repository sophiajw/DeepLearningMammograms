import os

import exercise_code.data_utils as data_utils

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'data')
txt_file = 'pre_processed_data.txt'
data = data_utils.load_mammography_data(os.path.join(dir_path, txt_file))