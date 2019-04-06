#!/usr/bin/env python
# coding: utf-8

# # Exemplo Deep Learning: Cancer
# 
# exemplo  de uma rede neural usada para identificar tumores maléficos e neutros
# 
# ## Funcionamento do Dataset
# 
# De acordo com a UCI Machine Learning Repository, o vetor de entrada é composto por:
# 
#     a) radius (mean of distances from center to points on the perimeter)
# 	b) texture (standard deviation of gray-scale values)
# 	c) perimeter
# 	d) area
# 	e) smoothness (local variation in radius lengths)
# 	f) compactness (perimeter^2 / area - 1.0)
# 	g) concavity (severity of concave portions of the contour)
# 	h) concave points (number of concave portions of the contour)
# 	i) symmetry 
# 	j) fractal dimension ("coastline approximation" - 1)
# 
# Sendo a saída:
# 
# 0 = tumor benigno
# 1 = tumor maligno
# 

# In[12]:


#Dataset a ser utilizado
from sklearn.datasets import load_breast_cancer
#Classicador Perceptron Multi Camadas
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
#preprocessador
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# In[6]:


#dataset
cancer_dataset = load_breast_cancer()

#imprimir os dados
print("\ndata\n")
print(cancer_dataset.data)
print("\ntargets\n")
print(cancer_dataset.target)


# In[8]:


#treinamento

X_train,X_test,y_train,y_test = train_test_split(cancer_dataset.data,cancer_dataset.target,test_size=50,random_state=0)


# In[9]:


#imprimir X_train e y_train

print("\nDados para treinamento\n")
print(X_train)
print("\nAlvos para treinamento\n")
print(y_train)


# In[14]:


#criar o scale
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)


# In[17]:


#numero de epocas
epochs = 4000
#numero de camadas
hl = [5,5,5]

#criar o classificador MultiPlayer Percetron
mlp = MLPClassifier(hidden_layer_sizes=hl,max_iter=epochs)


# In[18]:


#treinar a rede
mlp.fit(X_train_scale,y_train)


# In[19]:


#predicao e score
predict = mlp.predict(X_test_scale)
mlp.score(X_test_scale,y_test)


# In[20]:


#ver a confusion_matrix
print(confusion_matrix(y_test,predict)) 


# In[21]:


#por fim, ver os scores totais e a taxa de erro
print(classification_report(y_test,predict))


# In[ ]:




