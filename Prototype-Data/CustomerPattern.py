# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:12:02 2019

@author: MMOHTASHIM
"""

from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

style.use('ggplot')


def data_preprocessing(df):
    LE=LabelEncoder()
    df_1=df[["IP Address","Type of Business","City","Country"]]
    df_1 = df_1.fillna(df_1.mode().iloc[0])
    Preprocessed_data=[]
    for column in df_1.columns.tolist():
        Preprocessed_data.append(LE.fit_transform(df_1[column]))
    
    Preprocessed_data=np.array(Preprocessed_data)
    Final_Data=[]
    for customers in range(Preprocessed_data.shape[1]):
        Final_Data.append(Preprocessed_data[:,customers])
    
    df_preprocessed=pd.DataFrame(Final_Data,columns=["IP Address","Type of Business","City","Country"])
        
    
    return df_preprocessed

def Visual_Customer(df_preprocessed,labels):
    pca=PCA(n_components=2)
    reduced_data=pca.fit_transform(df_preprocessed)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    colors = {1:'red', 0:'blue'}
    df=pd.DataFrame(dict(labels=labels))

    
    ax.scatter(reduced_data[:,0],reduced_data[:,1],c=df["labels"].apply(lambda x:colors[x]))
    ax.set_title("Customer-Patter-PCA-Reduced")
    ax.set_xlabel("N-1")
    ax.set_ylabel("N-2")
    ax.grid(False)
    plt.show()
    
def ClusterDetection(df_preprocessed):
    pca=PCA(n_components=2)
    reduced_data=pca.fit_transform(df_preprocessed)
    
    MS=MeanShift()
    MS.fit(reduced_data)
    
    labels=MS.predict(reduced_data)
    return labels
    
    

if '__main__' == __name__:
    df=pd.read_excel("Clovitek.xlsx")
    df_preprocessed=data_preprocessing(df)
    labels=ClusterDetection(df_preprocessed)
    Visual_Customer(df_preprocessed,labels)
    