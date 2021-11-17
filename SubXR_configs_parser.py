import pandas as pd
import json

class SubXRParser():

    def __init__(self):
        with open(r"SubXR_configs_(Windows)", 'r') as f:
            self.configs = json.load(f)
    def __getitem__(self, key):
        if key in self.configs:
            return self.configs[key]
        else:
            []