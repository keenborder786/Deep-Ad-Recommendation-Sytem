# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:44:41 2019

@author: MMOHTASHIM
"""
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time
import os
import matplotlib.pyplot as plt
import pandas as pd

root_logdir = os.path.join(os.curdir, "my_logs")###tensoboard logs
def get_run_logdir():##helper function to store different models name with associated time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")  
    return os.path.join(root_logdir, run_id)

class Neural_network(object):
    ''''
    Neural Network Class
    '''
    
    def __init__(self,loss,learning_rate,l1,l2,saving_directory,decay):
        self.directory=saving_directory
        self.model=keras.models.Sequential()
        self.loss=loss
        self.optimizer= keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,  decay=decay)
        self.reg=keras.regularizers.l1_l2(l1=l1, l2=l2)
        self.history={}##history container for the performance of Neural Network
        
    def compile(self):##The main architecture and design
        self.model.add(keras.layers.Dense(300,input_shape=(1775,),kernel_regularizer=self.reg))###might have to change this
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(300,kernel_regularizer=self.reg))
        self.model.add(keras.layers.Dropout(0.3))
        
        self.model.add(keras.layers.Dense(400,activation="relu",kernel_regularizer=self.reg))
        self.model.add(keras.layers.Dropout(0.3))
        
        self.model.add(keras.layers.Dense(3,activation="softmax"))  
        
        
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'],)
            
        
        
        
    def fit(self,X_train,y_train,epoch,early_stopping=True):##Training Method, with the give parameters
        os.chdir(self.directory)##Model,logs and graphs will be saved in this directory
        
        run_logdir = get_run_logdir()##tensorboard log
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)##Tensorboad callback

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)#early stopping
    
        checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                    save_best_only=True) #checkpoint for saving the model
    
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.01)#creating validation dataset
        
        if early_stopping:
            history = self.model.fit(X_train, y_train, epochs=epoch,
                            validation_data=(X_valid, y_valid),
                            callbacks=[checkpoint_cb,tensorboard_cb,early_stopping_cb],verbose=1)
        else:
            history = self.model.fit(X_train, y_train, epochs=epoch,
                            validation_data=(X_valid, y_valid),
                            callbacks=[checkpoint_cb,tensorboard_cb],verbose=1)
        
        self.history=history##Different metric data contain in history
    
    def plot_performance(self,counter,save=False):##uses history a dict object to  plot the performance of neural network
        pd.DataFrame(self.history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.title(f"Result with following paramaters  Loss={self.loss}---Optimizer={self.optimizer}")
        plt.gca().set_ylim(0, 8) # set the vertical range 
        
        if save:
            run_time=time.strftime("run_%Y_%m_%d-%H_%M_%S")
            if not os.path.isdir("Graphs-Performance-Different Learning-Rates-Date{}".format(run_time)):
                os.mkdir("Graphs-Performance-Different Learning-Rates-Date{}".format(run_time))
            
            plt.savefig(self.directory+"\Graphs-Performance-Different Learning-Rates-Date{}".format(run_time)+f"\Graph---{counter}.png")
        
        
