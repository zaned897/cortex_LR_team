"""Firt version to evaluate the NLP model and the 
   Natural Language Search based on the info search
   Algorithms 

   version:  0.5
   @author: Eduardo Santos, Asymm Developers
   date: Jan 2021
"""

#%%                              LOAD DEPENDENCIES

import spacy
import numpy as np 
from pdf2image import convert_from_path
from nltk.stem.porter import *
import nltk 
import pytesseract as pt
from pytesseract import Output
import cv2
import concurrent.futures
import lossrun
from configobj import ConfigObj
import matplotlib.pyplot as plt
import collections

print('Loading dependnecies...')





#%%                     LOAD DATA
#hard code file 
FILE = '/home/zned897/Proyects/pdf_text_extractor/nowinsurance-loss-runs/docs/lossruns_feasibility/MultipleClaims3.pdf'
MODEL_PATH = '/home/zned897/Proyects/pdf_text_extractor/NPDB_ner_model'
print('Reading file: ...' + str(FILE[-12:]))

# load the data
images = convert_from_path(FILE, grayscale=True, dpi=350)
print('The image size is: ' + str(np.array(images[0]).shape))


#appy image enhanced if necessary (typically a blur works fine)

#multi-threads
def main_ocr(image):
    image = np.array(image)
    image[image<=125]=0
    image[image>126] = 255
    image = cv2.blur(image, (2,2))
    return pt.image_to_data(image, output_type=Output.DICT)


#apply ocr in dictionary format #pre-proc the text applying tokenizer and  stem if needed #create sentences #label the ner in sentences using the big brother model #use the data to retrain the little broter model based on
print('Applying OCR in multi threads in ' + str(len(images))+ ' pages')
dictionaries = []

"""with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    results = [executor.submit(main_ocr, image) for image in images]
    for result in concurrent.futures.as_completed(results):
        dictionaries.append(result.result())
"""
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:  # add threads as max worker when sagemaker up
   results = executor.map(main_ocr,images)     # return the process in order 
   [dictionaries.append(result) for result in results]






#%                        SET THE CONFIGURATIONS

print('Loading the configuration files...')

try:
    nlp('model?')
except:
    print('loading NPDB NER model...')
    #print(MODEL_PATH)
    nlp = spacy.load(MODEL_PATH)
#nlp = spacy.load('/home/zned897/Proyects/pdf_text_extractor/external_models/NPDB_models/NPDB_ner_model')


page = 0
# load the configuration files and declare grammar rules 
TOPICS = ConfigObj('./config/config_topic.ino')     # load the interest points
ENTS = ConfigObj('./config/config_ents.ino')      # load the NAME ENTITY RULES 
open_exp = r"\b[A-Z][A-Z]+\b"                               # reg exp rules                
mon_exp = r"[^0-9\.0-9]+"
alpha_exp = r"(?=[A-Za-z])\w+.*(?=[a-z])\w+."
# 
suspects = lossrun.search_rules(dictionaries[page], TOPICS)  # store the suspects (topic fit)
spatial_filter_ver, tops, spatial_filter_hor, lefts = lossrun.spatial_filter(dictionaries[page], suspects, 'LOSSRUN') 

#print(suspects)
pass
black_flag = True
claims = []


for topic in range(len(suspects)):

  
  sentence_ver  = ' '.join(spatial_filter_ver[topic])
  sentence_hor  = ' '.join(spatial_filter_hor[topic])
  sentence_ver = re.sub(r"\s+",' ', sentence_ver)
  sentence_hor = re.sub(r"\s+",' ', sentence_hor)

  doc_hor = nlp(sentence_hor)

  if suspects[topic][0] == 'claim' and black_flag:
       sf = []
       [sf.append(word) for word in spatial_filter_ver[topic] if word != '']
       [claims.append(word) for word in sf if lossrun.isaclaim(word)]
       black_flag = False

"""
  for ent in doc_hor.ents:
    if suspects[topic][0] == 'insured':
        if ent.label_ in Ents['insured']:
            print(str(suspects[topic][0]) + ': ' + str(ent.text) + ' ' + str(ent.label_))
          #  break
    if suspects[topic][0] == 'as_of':
       if ent.label_ in Ents['as_of']:
            print(str(suspects[topic][0]) + ': ' + str(ent.text) + ' ' + str(ent.label_))
           # break
"""
img = np.array(images[page])
coords = []
#[print(word) for word in dictionaries[page]['text'] if word in claims]
for i, claim, in enumerate(dictionaries[page]['text']):
    if claim in claims:
        top = dictionaries[page]['top'][i]
        left  = dictionaries[page]['left'][i]
        width  = dictionaries[page]['width'][i]
        height  = dictionaries[page]['height'][i]
        coords.append(([top, left, width, height]))
        cv2.line(img,(left,top),(max(dictionaries[page]['left']), top),(0,0,0), 10)

coords.append(([max(dictionaries[page]['top']), 0, 0, 0]))
claims
# %%
plt.figure(figsize=(17,15))
plt.imshow(img, cmap='gray')
#%%                          TESTING CELL IGNORE IT 

stus = []
results = []
for i, ENT in enumerate(suspects):
    #print(i, topic[0])
    #print('........')
    sent_hor  = nlp(' '.join(spatial_filter_hor[i]))
    sent_ver =  nlp(' '.join(spatial_filter_ver[i]))

    if ENTS[ENT[0]] == []:
        aux = re.findall(open_exp, ' '.join(spatial_filter_hor[i]))
        sentence = ' '.join(aux)
        #print(ENT[0] + ':' + str(sentence))
        results.append([sentence, suspects[i][-1]])
    
    elif ENT[0] == 'status':
        [results.append([word, tops[i][k]]) for k, word in enumerate(spatial_filter_ver[i]) if 'CL' in word.upper()]
        [results.append([word, tops[i][k]]) for k, word in enumerate(spatial_filter_ver[i]) if 'OP' in word.upper()]
        #print(stus)
        #results.append([stus, suspects[i][-1]])
    elif '-' in ENTS[ENT[0]]:
        pass
    [results.append([ent.text, suspects[i][-1]]) for ent in sent_hor.ents if ent.label_ in ENTS[ENT[0]]]
    [results.append([ent.text, suspects[i][-1]]) for ent in sent_ver.ents if ent.label_ in ENTS[ENT[0]]]

#    for ent in sent_ver.ents:
#        if ent.label_ in ENTS[ENT[0]]:
#            print(ent.text, ent.label_)

results
outputs = collections.defaultdict(list) 
#%%
for i, result in enumerate(results):
    x1 = result[1] + coords[0][1]//2
    if x1 <= coords[0][0]:
        print(result[0], 'Cover info')
        outputs['cover'].append(result[0])
       
    for j in range(len(coords)-1):
        if coords[j][0] < x1 < coords[j + 1][0]:
            print(result[0], claims[j])
            outputs[claims[j]].append(result[0])

#%%         TESET CELL IGNORE IT 
""""

import copy
import string

#claim_test = '3:37:26'
claim_test = 'sadkhsa-akshdk'
punkt = string.punctuation.replace('-','').replace('.','')

len(claim_test) >= 7 and any(char.isdigit() for char in claim_test) and not any(char in punkt for char in claim_test)
"""
#%%  EXTRACT THE CLAIMS AND POILICY NUMBERS

import lossrun
import numpy as np
gnb = lossrun.get_NB()

gnb.predict(np.array([[396,220,198,6,7,3,0,0]]))