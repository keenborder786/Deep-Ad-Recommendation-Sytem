# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:14:20 2019

@author: MMOHTASHIM
"""
import os
os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype\src")
from SpeechtoText import Flac_Converter
from SpeechtoText import Speech_Converter
from AdRecommendation import Neural_network

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV,StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from tensorflow import keras
from gensim.models import Word2Vec
from statistics import mean

def Flacc_Converter_Test(): ##How User will use my converter-one case-the time complexity is really high?
    ''''
    This function was written to test a back-end library which converts a bunch of mp4 files 
    into flacc file which is better for speech recognization.
    
    '''
    categories=["Food","Sports","Technology"]
    for category in tqdm(categories):
        os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData") #the directory will be change according to your choice
        
        for company in os.listdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-Mp4Scrapped-Data\{}".format(category)):
            
            for file in os.listdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-Mp4Scrapped-Data\{}\{}".format(category,company)):
    
                    flacc_converter=Flac_Converter(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-Mp4Scrapped-Data\{}\{}".format(category,company),file)
                    
                    if not os.path.isdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}".format(category)):
                        os.makedirs(f"{category}")
                   
                    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}".format(category))
                    
                    if not os.path.isdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}\{}".format(category,company)):
                        os.makedirs(f"{company}")
                    
                    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}\{}".format(category,company))            
                    flacc_converter.convert()



def Speech_Converter_Test():##How will user use my speech recognization library-one case-the time complexity is really high?
    ''''
    This function was wrritten to test speech recognization back-end library and convert a bunch
    of flacc files into text and store it as an array with the desired category of the ad
    Please note the data stored in the array is in raw form-need to further process it...
    
    '''
    
    categories=["Food","Sports","Technology"]
    Final_Array_data=[]
    labels=[]
    for category in tqdm(categories):
        
        for company in os.listdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}".format(category)):
            
            for file in os.listdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}\{}".format(category,company)):
    
                    Speech_Watson=Speech_Converter(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-FlaccData\{}\{}".format(category,company),file) #this line of code cam take time,as it talks directly to cloud
                    data=Speech_Watson.store_in_array()
                    
                    Final_Array_data.append(data)
                    labels.append(category)
    
    Final_Array=np.array(Final_Array_data)
    labels=np.array(labels)
    
    
    np.save(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\Final_Array.npy",Final_Array)
    np.save(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\labels.npy",labels)
    
def one_hot_encode(y):
    ''''
    helper function to one hot encode the given input
    '''
    
    unique_categories=["Food","Sports","Technology"]
    
    y_final=[]
    for value in y:
        one_hot=np.zeros(len(unique_categories))
        index=unique_categories.index(value)
        one_hot[index]=1
        y_final.append(one_hot)
    
    return y_final

def word_embedding(new=False):##Using Word Embedding
    ''''
    This function is used to create word embedding for the given array of text
    
    '''
    Final_Array_data=np.load(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\Final_Array.npy",allow_pickle=True)

    labels=np.load(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\labels.npy",allow_pickle=True)    
    
    Combined_Data=list(zip(Final_Array_data,labels))
    X_raw=[]
    y_raw=[]
    X_embedded=[]
    
    for data in Combined_Data:##Making use of raw data and breaking up the corpus into sentences and associating the desired label with the given sentence
        for sentence in data[0]:
            X_raw.append(sentence)
            y_raw.append(data[1])

    if new:
        model = Word2Vec(X_raw, size=100, window=5, min_count=1, workers=4)
        model.save("word2vec.model")
    else:
        model = Word2Vec.load(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\word2vec.model")
        for sentence in X_raw:
            embedding=[]
            for word in sentence:
                vector = model.wv[word]
                embedding.append(vector)
            X_embedded.append(np.mean(embedding,0))
            
    X_embedded=np.array(X_embedded)
    y=np.array(one_hot_encode(y_raw))

    permutation = list(np.random.permutation(X_embedded.shape[0]))##randonly shuffling the matrix
    X_embedded = X_embedded[permutation,:]
    y = y[permutation,:]
    
    
    return X_embedded,y
                
                
                
    
    
     
                
        
def preprocessing_pipeline(feature_extraction):##Using TDIDF or CountVectorizer
    ''''
    This function would load the raw array of data and associate label ,then
    preprocess it in order to make it excuetable and understandable for Neural Network.

    '''
    
    Final_Array_data=np.load(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\Final_Array.npy",allow_pickle=True)

    labels=np.load(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\labels.npy",allow_pickle=True)    
    
    Combined_Data=list(zip(Final_Array_data,labels))
    X_raw=[]
    y_raw=[]

    for data in Combined_Data:##Making use of raw data and breaking up the corpus into sentences and associating the desired label with the given sentence
        for sentence in data[0]:
            X_raw.append(sentence)
            y_raw.append(data[1])
            

    if feature_extraction=="CountVectorizer":##Converting the Raw Text to numerical conversion by using the standard text feature extraction methods
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X_raw)
        ##pickling the vectorizer
        with open(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\vectorizer.pickle', 'wb') as handle:
            pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif feature_extraction=="TfidfVectorizer":
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X_raw)
        ##pickling the vectorizer
        with open(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\vectorizer.pickle', 'wb') as handle:
            pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    y=np.array(one_hot_encode(y_raw))##converting the categorical labels into onehot encoded array
    X=X.toarray()##Converting the sparse matrix to array
    
    permutation = list(np.random.permutation(X.shape[0]))##randonly shuffling the matrix
    X = X[permutation,:]
    y = y[permutation,:]
    
    
    return X,y
#Only need to use on approach from TD-IDF,WordEmbedding or CountVectorizer  

def Beta_test_neural_network(X,y,saving_directory,l1,l2,decay,learning_rate,loss="categorical_crossentropy",epoch=20,early_stopping=True):   
    ''''
    This function is to use to test the back-end Neural Network made-this is just
    a function to run the back-end code of the Neural Network Archtecture.
    '''
    
    Ad_Recommendation=Neural_network(loss,learning_rate,l1,l2,saving_directory,decay)
    
    Ad_Recommendation.compile()
    
    Ad_Recommendation.fit(X,y,epoch,early_stopping)
        
    return Ad_Recommendation

def Predict_new_audio(model_name,vectorize):
    '''''
    
    This function is only for Testing Purposes and takes one audio file ,make it go through the 
    same training pipeline and predict the different possible categories that our neural network 
    has predicted
    
    '''
    
    flacc_converter=Flac_Converter(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\validation","test(2).mp4")
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data")
    flacc_converter.convert()##Converting the audio file into flac file
    
    Speech_Watson=Speech_Converter(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data",r"test(2).mp4.flac")
    
    data=Speech_Watson.store_in_array()
    data=np.array(data)
    
    data=data.reshape(data.shape[0],)###Storing the Audio file into chunks of sentences identified by cloud ibm watson
    if vectorize==True:
        with open('vectorizer.pickle', 'rb') as handle:##using the pre-trained vectorizer
            vectorizer = pickle.load(handle)
            
        data=vectorizer.transform(data)
        data=data.toarray()
        model=keras.models.load_model(f"{model_name}")#Loading the pretrained model
        predicted=model.predict(data)
        unique_categories=["Food","Sports","Technology"]##Unique categories
        print("The categories predicted are as following in the given clip:",[unique_categories[label] for label in np.argmax(predicted,1)])
    else:
        X_embedded=[]
        model = Word2Vec.load(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Data\word2vec.model")
        for sentence in data:
            embedding=[]
            for word in sentence:
                vector = model.wv[word]
                embedding.append(vector)
            X_embedded.append(np.mean(embedding,0))
        X_embedded=np.array(X_embedded)
        print(X_embedded.shape)
        model=keras.models.load_model(f"{model_name}")#Loading the pretrained model
        predicted=model.predict(X_embedded)
        unique_categories=["Food","Sports","Technology"]##Unique categories
        print("The categories predicted are as following in the given clip:",[unique_categories[label] for label in np.argmax(predicted,1)])
    

def create_model( nl1=1, nl2=1,  nl3=1, 
                 nn1=1000, nn2=500, nn3 = 200, lr=0.01, decay=0., l1=0.01, l2=0.01,
                act = 'relu', dropout=0, input_shape=1475, output_shape=3):
    
    '''This is a model generating function so that we can search over neural net 
    parameters and architecture--just a backend function for hyperparameter ,the result of this randomsearchCV helps 
    to define the back-end Arcihtecture'''
    
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,  decay=decay)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
                                                     
    model = keras.models.Sequential()
    
    # for the firt layer we need to specify the input dimensions
    first=True
    
    for i in range(nl1):
        if first:
            model.add(keras.layers.Dense(nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        
        else: 
            model.add(keras.layers.Dense(nn1, activation=act, kernel_regularizer=reg))
        
        if dropout!=0:
            model.add(keras.layers.Dropout(dropout))
            
    for i in range(nl2):
        
        if first:
            model.add(keras.layers.Dense(nn2, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        
        else: 
            model.add(keras.layers.Dense(nn2, activation=act, kernel_regularizer=reg))
        
        if dropout!=0:
            model.add(keras.layers.Dropout(dropout))
            
    for i in range(nl3):
        if first:
            model.add(keras.layers.Dense(nn3, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        
        else: 
            model.add(keras.layers.Dense(nn3, activation=act, kernel_regularizer=reg))
        
        if dropout!=0:
            model.add(keras.layers.Dropout(dropout))
            
    model.add(keras.layers.Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'],)
    
    return model
   
    
    
    
    

if '__main__' == __name__:
   ###########################################
#   X,y=word_embedding(new=False)
   X,y=preprocessing_pipeline("CountVectorizer")
   
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    ######################################

   
    #################################################
    
    ##RANDOMSEARCH-cv-hYPERMATER tUNING FOR BACKEND ARCHTECTURE
#   # model class to use in the scikit random search CV 
#   model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=6, batch_size=20, verbose=1)
#   # learning algorithm parameters
#   lr=[1e-2, 1e-3, 1e-4]
#   decay=[1e-6,1e-9,0]
#    
#   # activation
#   activation=['relu', 'sigmoid']
#    
#   # numbers of layers
#   nl1 = [0,1,2,3]
#   nl2 = [0,1,2,3]
#   nl3 = [0,1,2,3]
#    
#   # neurons in each layer
#   nn1=[300,700,1400, 2100,]
#   nn2=[100,400,800]
#   nn3=[50,150,300]
#    
#   # dropout and regularisation
#   dropout = [0, 0.1, 0.2, 0.3]
#   l1 = [0, 0.01, 0.003, 0.001,0.0001]
#   l2 = [0, 0.01, 0.003, 0.001,0.0001]
#    
#   # dictionary summary
#   param_grid = dict(
#                        nl1=nl1, nl2=nl2, nl3=nl3, nn1=nn1, nn2=nn2, nn3=nn3,
#                        act=activation, l1=l1, l2=l2, lr=lr, decay=decay, dropout=dropout, 
#                        input_shape=[X_train.shape[1]], output_shape = [y_train.shape[1]],
#                     )
#   
#   
#   grid = RandomizedSearchCV(estimator=model, cv=3, param_distributions=param_grid, 
#                          verbose=20,  n_iter=10, n_jobs=-1)
#   
#   grid_result = grid.fit(X_train, y_train)
#   ############################################################
#   
#


#   CHECKING THE PERFORMANCE OF MY FINAL ARCHITECTURE
   val_loss=[]
   for i in tqdm(range(2)):##trials for hyperparameter tuning-random search rather specfic grid search
       Ad_Recommendation=Beta_test_neural_network(X,y,l1=0.0001,l2=0,decay=0,learning_rate=0.01,loss="categorical_crossentropy",epoch=40,saving_directory=r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data")##Optimizer will be Adam by defeault
       
       val_loss.append(mean(Ad_Recommendation.history.history["val_loss"]))
   
       Ad_Recommendation.plot_performance(i,save=True)    
       
       Predict_new_audio("my_keras_model.h5",True) 
   
   print(f"The mean val loss for each trial is {val_loss}")
   
   ###################################################################
#    
#    Flacc_Converter_Test()
#     Speech_Converter_Test()
  
     
       

  
        


         