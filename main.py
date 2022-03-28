import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

#########Find file dataset.csv###################
dataset_credit = pd.read_csv('Datasets/credit_risk_dataset.csv')

##############Data processing##############
#Null values
#Removendo idades inválidas
dataset_credit2 = dataset_credit.drop(dataset_credit[dataset_credit['age'] <= 17].index)

#Verificando dados nulos.
print(dataset_credit2.loc[pd.isnull(dataset_credit2['loan'])]) #printa o valor NaN
dataset_credit2['loan'].fillna(dataset_credit2['loan'].mean(), inplace = True) #Corrige o valor NaN
print(dataset_credit2.loc[pd.isnull(dataset_credit2['loan'])]) #printa o valor NaN

#Previsores e Divisores.
x_dtCredit = dataset_credit2.iloc[:, 1:4].values
y_dtCredit = dataset_credit2.iloc[:, 4].values

print(f'\nX_credit = {x_dtCredit}\n\nY_credit = {y_dtCredit}\n\n')

#padronizando os dados.
scale_credit = StandardScaler()
x_dtCredit = scale_credit.fit_transform(x_dtCredit)

#base de treinamento.
x_dtCredit_treinamento, x_dtCredit_teste, y_dtCredit_treinamento, y_dtCredit_teste = train_test_split(x_dtCredit, y_dtCredit, test_size = 0.25, random_state = 0)

#salvando dados em um arquivo.
with open('Datasets/credit.pkl', mode = 'wb') as f:
    pickle.dump([x_dtCredit_treinamento, y_dtCredit_treinamento, x_dtCredit_teste, y_dtCredit_teste], f)

####Naive Bayes#####
#abrindo o arquivo com as informações salvas anteriormente
with open('Datasets/credit.pkl', 'rb') as f:
    x_dtCredit_treinamento, y_dtCredit_treinamento, x_dtCredit_teste, y_dtCredit_teste = pickle.load(f)

naive_dtCredit_data = GaussianNB()
naive_dtCredit_data.fit(x_dtCredit_treinamento, y_dtCredit_treinamento)

prevision_dtCredit_data = naive_dtCredit_data.predict(x_dtCredit_teste)

print(prevision_dtCredit_data)
print(y_dtCredit_teste)
print("\n\n#--- Precisão ---#")
print(f'{accuracy_score(y_dtCredit_teste, prevision_dtCredit_data) * 100}%')
print(classification_report(y_dtCredit_teste, prevision_dtCredit_data))
