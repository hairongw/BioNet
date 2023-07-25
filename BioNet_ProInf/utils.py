import json 
import os 
import pandas as pd 
import numpy as np
import torch 
import pickle 
import json 
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score

def read_params():
        """
        reading parameters
        """
        params = dict()
        with open("params.json") as f:
            params = json.load(f)

        return params 

class DotDict(object):
    """
    convert python dictionary into dot the one that support "." operation
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)
