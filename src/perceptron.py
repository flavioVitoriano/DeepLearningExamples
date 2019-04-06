#!/usr/bin/env python
# coding: utf-8

# ## Exemplo de perceptron
# 
# exemplo bem simples do funcionamento de um perceptron com a função degrau
# 
# __Codigo baseado no algoritmo de Thomas Countz__
# Fonte: https://gist.githubusercontent.com/Thomascountz/77670d1fd621364bc41a7094563a7b9c/raw/7f2d52dc8b879a2454467bb7acdee31c4c3e3d41/perceptron.py
# 
# Exemplo de um AND e OR com um perceptron simples
# 
# OBS: Nesse exemplo, o valor 1 (um) será tido como verdadeiro enquanto o valor 0 (zero) será tido como falso
# 

# In[1]:


# Perceptron de Thomas Countz

import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


# ## Criação do dataset de treinamento
# 

# In[2]:


samples = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([0,1]),
    np.array([1,1]),
]

and_labels = np.array([0,0,0,1])
or_labels = np.array([0,1,1,1])


# ## Criação dos perceptrons

# In[3]:


# perceptron com um vetor de duas entradas
and_perceptron = Perceptron(2)
or_perceptron = Perceptron(2)


# ## Treinamento

# In[4]:


and_perceptron.train(samples,and_labels)
or_perceptron.train(samples,or_labels)


# ## Ver a predição

# In[6]:


#AND
print("AND neural\n1 e 1\n1 e 0\n0 e 1\n0 e 0\n")
print(and_perceptron.predict(np.array([1,1])))
print(and_perceptron.predict(np.array([1,0])))
print(and_perceptron.predict(np.array([0,1])))
print(and_perceptron.predict(np.array([0,0])))

#OR
print("\nOR neural\n1 e 1\n1 e 0\n0 e 1\n0 e 0\n")
print(or_perceptron.predict(np.array([1,1])))
print(or_perceptron.predict(np.array([1,0])))
print(or_perceptron.predict(np.array([0,1])))
print(or_perceptron.predict(np.array([0,0])))


# In[ ]:




