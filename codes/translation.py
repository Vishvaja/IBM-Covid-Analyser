# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:29:57 2020

@author: pjai1
"""

from translate import Translator
translator= Translator(to_lang="English")
translation = translator.translate("iravu vanakkam")
print (translation)



from googletrans import Translator
translator = Translator()

import pandas as pd
df_es=pd.read_csv('sol.csv',nrows=100000,engine='python')

# print first rows of the data frame
df_es.head()

# make a deep copy of the data frame
df_en = df_es.copy()

# translate columns' name using rename function
df_en.rename(columns=lambda x: translator.translate(x).text, inplace=True)

# translated column names
df_en.columns
                        
#Diff code
translation={}
X=df_es.iloc[:,0].values
X=X.tolist()
for i in range(100000):
    translation[i]=translator.translate(X[i]).text

df_es['text']=translation




translations = {}
for column in df_en.columns:
    # unique elements of the column
    unique_elements = df_en[column].unique()
    for element in unique_elements:
        # add translation to the dictionary
        translations[element] = translator.translate(element).text
    
print(translations)



translation = translator.translate("Je suis")
print(translation)
