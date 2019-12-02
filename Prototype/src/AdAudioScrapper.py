# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:59:10 2019

@author: MMOHTASHIM
"""

import os
from bs4 import BeautifulSoup as bs
import requests
from pytube import YouTube
import pytube
from tqdm import tqdm


class Audio_Ad_Scrapper(object):##used to download Audio from youtube videos and save them as mp4 files
    '''' 
    
    Main Class to download videos from youtube
    
    '''
    
    def __init__(self,Category,filepath):#Class Intiation
        self.base="https://www.youtube.com/results?search_query="
        self.Category=Category##Type of Ad
        self.links=[]###Video Link
        self.filepath=filepath
    def create_links(self):
        '''
        Input Self
        
        Description This Method Generates all links of the main search query page and append it to
        self.links
        
        Returns Nothing
        '''
        r = requests.get(self.base+self.Category)
        page = r.text
        soup=bs(page,'html.parser')
        vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})
        for v in vids:
            tmp = 'https://www.youtube.com' + v['href']
            self.links.append(tmp)
        
        
    def download_audio(self):
        ''''
        Input Self 
        Description=This Method downloads audio directly from youtube video and store into a
        mp4-audio file for each video
        
        Returns Nothing
        '''
        count=0
        os.makedirs(r"{}\{}".format(self.filepath,self.Category))
        for vid in tqdm(self.links[:8]):
            try:
                # increment counter:
                count+=1
             
                # initiate the class:
                yt = YouTube(vid)
                        
                # grab the audio:
                stream = yt.streams.get_by_itag('140')
                # set the output file name:
                filename='Audio'+"{}".format(self.Category)+str(count)                
                # download the video:
                stream.download(r"{}\{}".format(self.filepath,self.Category),filename=filename)
                        
            ##Error Catching
            except pytube.exceptions.RegexMatchError:
                print("Error-Could not match the file")
            except pytube.exceptions.VideoUnavailable:
                print("Error-Could not find the file")
            except pytube.exceptions.ExtractError:
                print("Error-Could not download the file")
            except pytube.exceptions.LiveStreamError:
                print("Error-Could not Stream the file")
            except:
                print("Unknown Error")



  
if '__main__' == __name__:
        S=Audio_Ad_Scrapper("nike+radio+ads",r"C:\Users\MMOHTASHIM\Anaconda3\libs\Prototype-Data\Audio-Mp4Scrapped-Data")
        S.create_links()
        S.download_audio()
