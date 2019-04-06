#!/usr/bin/env python
# coding: utf-8

# # Exemplo de Deep Learning usando o dataset Iris
# 
# Exemplo usando o Dataset Iris. Determina a classe de uma flor de íris (**setosa**, **versicolor** ou **virginica**)
# Pelas dimensões da sépala (_sepal_) e  da pétala (_petal_) usando o Multi-Layer Perceptron.
# 
# 
# 
# Importaremos os módulos usados no exemplo, que serão o **pandas**, **scikit-learn** e o **numpy**

# In[2]:


#importar as bibliotecas
import pandas as pd
import numpy as np
#classificador
from sklearn.neural_network import MLPClassifier
#treinamento
from sklearn.model_selection import train_test_split
#preprocessamento
from sklearn.preprocessing import StandardScaler
#dataset
from sklearn.datasets import load_iris
#matriz
from sklearn.metrics import confusion_matrix, classification_report


# ## Dataset
# 
# Nota: Utilizarei o dataset do UC _Irvine Machine Learning Repository_ 

# In[3]:


#colunas do dataset
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#criar dataset
iris_dataset = load_iris()


# mostrar os exemplos e a saída do dataset

# In[4]:


iris_dataset.data


# In[5]:


iris_dataset.target


# ## Treinamento
# 
# 
# 
# Gerar as variaveis de teste e treinamento
# 
# **Usando o dataset iris**
# 
# test_size = 45 por cento
# 
# random_state = 0
# 

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data,iris_dataset.target, test_size=45, random_state=0)


# In[7]:


X_train


# In[15]:


y_train


# ## Pre-processamento

# In[9]:


scaler = StandardScaler()

scaler.fit(X_train)

#criar as scalações no preprocessamento
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)


# ## Criação da rede
# 
# epocas = 5000 iterações
# 
# camadas ocultas = 3 camadas
# 
# 
# 1 camada = 8 neurônios
# 
# 2 camada = 10 neurônios
# 
# 3 camada = 8 neurônios

# In[10]:


#numero de epocas para a atualização dos pesos
epochs = 5000
#definit o tamanho das camadas ocultas do Perceptron multicamadas
hidden_layers = [8,10,8]

#criação da rede MultiLayerPerceptron
mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=epochs)


# ## Realizar o Treinamento

# In[11]:


mlp.fit(X_train_scale,y_train)


# ## Predição e score

# In[12]:


predict = mlp.predict(X_test_scale)
mlp.score(X_test_scale,y_test)


# In[13]:


print(confusion_matrix(y_test,predict)) 


# In[14]:


print(classification_report(y_test,predict)) 

