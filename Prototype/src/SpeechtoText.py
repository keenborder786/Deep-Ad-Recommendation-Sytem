# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:29:03 2019

@author: MMOHTASHIM
"""

import speech_recognition as sr
from os import path
from pydub import AudioSegment
from ibm_watson import SpeechToTextV1
import os

AudioSegment.converter = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
API_KEY="NL-w11y8plDNU17sCGRSPNBC3le1j9vFuGnLS0AYidEZ"
URL="https://stream.watsonplatform.net/speech-to-text/api"



class Flac_Converter():
    ''''
    Convert Mp4 files collected from youtube to Flacc format which will be used later on to 
    convert to text file.
    
    Flac Format is preffered because it is far more compressed which saves alot of data
    
    '''
    def __init__(self,filepath,filename):
        self.filename=filename
        self.filepath=filepath
        self.mp4_version=AudioSegment.from_file("{}\{}".format(self.filepath,self.filename),"mp4")#Orginal MP4 file
    def convert(self):
        self.mp4_version.export("{}.flac".format(self.filename), format="flac")#Using ffmpeg coding to covert to flac format


class Speech_Converter():
    ''''
    Beta Implementation of Converting Speech to Text
    
    This use the highly-trained ibm watson to convert a flac file to text and then store into and array which
    we can later on use as final input on our neural network-after converting them to numerical values
    
    '''
    
    def __init__(self,filepath,filename):
        self.filepath=filepath
        self.filename=filename
        self.converter=SpeechToTextV1(
            iam_apikey =API_KEY,
            url = URL
        )##ibm watson cloud service-trial version-can be bought
        
    def convert(self):
        with open("{}\{}".format(self.filepath,self.filename), 'rb') as audio_file:
            speech_recognition_results = self.converter.recognize(
                audio=audio_file,
                content_type='audio/flac').get_result()##Opening the flacc and covnerting to the desired String Format
        return speech_recognition_results
    
    def store_in_array(self):
        data=[]
        speech_recognition_results=self.convert()##Recalling the convert method which return the text result from the given flacc file
        for i in speech_recognition_results["results"]:
            speech=i["alternatives"][0]["transcript"]
            data.append(speech)##storing different parts of text into a single array
        return data

            
            
            
if '__main__' == __name__:
    converted=Flac_Converter(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\test",r"Audionike+radio+ads1.mp4")
    converted.convert()
    
    
    
    