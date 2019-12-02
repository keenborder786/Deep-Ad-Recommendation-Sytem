# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 00:05:22 2019

@author: MMOHTASHIM
"""

from setuptools import setup

setup(name="AdRecommendation",
      version="0.0.1",
      description="Prototype Version of Ad-Recommendation",
      py_modules=["AdAudioScrapper","AdCaptionScrapper","SpeechtoText",
                  "AdRecommendation","PlaylistScrapper"],
                  package_dir={'':'src'},
                  classifiers=['Programming Language :: Python :: 3.7',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.5',
                               'License :: OSI Approved :: Artistic License']
                               )



