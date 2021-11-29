import os
import shutil
import numpy as np
# mkdirs = lambda x: os.path.exists(x) or os.makedirs()
# rmdirs = lambda x: (not os.path.exists(x)) or shutil.rmtree(x)
remove_and_create = lambda x: (not shutil.rmtree(os.path.abspath(x), ignore_errors=True)) and os.makedirs(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)
dir_content = lambda path, random: np.random.permutation(os.listdir(path)) if random else sorted(os.listdir(path))
size_dir_content = lambda path: len(os.listdir(path))
