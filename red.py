# Importamos las librerías necesarias para la construcción de la red
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        print("Se inicia el entrenamiento de pesos...")
        num_data =  len(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # Se inicializan los pesos
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Se define la regla de Hebb
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Matriz de pesos de diagonal nula
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
    
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print("Se inicia la comparación...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        copied_data = np.copy(data)
        
        # Se define la lista de predicciones
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def _run(self, init_s):
        if self.asyn==False:
            """
            Actualización síncrona
            """
            # Se computa la energía inicial
            s = init_s

            e = self.energy(s)
            
            # Se itera
            for i in range(self.num_iter):
                # Se actualiza la energía
                s = np.sign(self.W @ s - self.threshold)
                # Se asigna la energía actual
                e_new = self.energy(s)
                
                # Si la veriable enrgía converge, se devuelve el valor actual
                if e == e_new:
                    return s
                # Se actualiza la variable energía
                e = e_new
            return s
        else:
            """
            Actualización asíncrona
            """
            # Se computa la energía inicial
            s = init_s
            e = self.energy(s)
            
            # Se itera
            for i in range(self.num_iter):
                for j in range(100):
                    # Se elige una neurona aleatoriamente
                    idx = np.random.randint(0, self.num_neuron) 
                    # Se actualiza la energía
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                
                # Se asigna la energía actual
                e_new = self.energy(s)
                
                # Si la veriable enrgía converge, se devuelve el valor actual
                if e == e_new:
                    return s
                # Se actualiza la variable energía
                e = e_new
            return s
    
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)