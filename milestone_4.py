import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import re
import urllib.request
import os
import jinja2

import keras
from keras.models import load_model

def predict_torrmow():
    model = load_model('model.h1')

    df = pd.read_csv("C:\\Users\\User\\Desktop\\data_mining\\goldprice_25-May-2020 14-14-48.csv")
    