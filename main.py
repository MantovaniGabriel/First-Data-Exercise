import numpy as np
import plotly.express as pl #lib for graphics.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #lib for graphics interface.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

#########Find file dataset.csv###################
dataset_credit = pd.read_csv('Datasets/dataset.csv')

##############Data processing##############
#Null values
#Removendo idades inválidas
dataset_credit2 = dataset_credit.drop(dataset_credit[dataset_credit['age'] <= 17].index)

#Removendo colunas não uteis para o uso.
dataset_credit2 = dataset_credit2.drop('special disease', axis = 1)
dataset_credit2 = dataset_credit2.drop('religion', axis = 1)
dataset_credit2 = dataset_credit2.drop('break in love', axis = 1)
dataset_credit2 = dataset_credit2.drop('smoke', axis = 1)

#Verificando dados nulos.
print(dataset_credit2.isnull())

#Previsores e Divisores.

#print(dataset_credit2)