"""Code for the lossrun info extraction demo
   temporal code that extract info in lossrun reports 

    1.- Apply text rules for claims and policies extraction 
        a) the word must contain numbers
        b) muste len(word)>=7
        c) can contain only (.), (-), and (_) as punctuation chars
    2.- A Naive Bayes classificator is applies based on the claims, policies frecuency. space
        distribution, and chars content
    3.- Extract info ruled in the config files (topics and entities) 
    4.- Group the info to the claims (in between claims)

date: Feb 2021
@author: Eduardo S. Romulo T. Raul V. Alberto de O. Asymm Developers
"""


#%%                             LOAD DEPENDENCIES
import re
import lossrun

from numpy import array
from spacy import load
from matplotlib.pyplot import figure
from collections import defaultdict
from matplotlib.pyplot import imshow
from pdf2image import convert_from_path
from pytesseract import image_to_data
from pytesseract import Output
from cv2 import blur
from cv2 import line
from concurrent.futures import ThreadPoolExecutor
from configobj import ConfigObj
from multiprocessing import cpu_count
from time import time

print("\tLossrun Reports Info Extraction\nDemo version by Asymm Developers & and Now Insurance\ndate: Feb 2021")
print('.'*50 + '\n'*3)




#%                         LOAD DATA
# hardcode file 
FILE = '/home/zned897/Proyects/pdf_text_extractor/nowinsurance-loss-runs/docs/lossruns_feasibility/MultipleClaims3.pdf'
MODEL_PATH = '/home/zned897/Proyects/pdf_text_extractor/nowinsurance-loss-runs/models/lossruns_models/model_LRL01'

# time the pdf 2 image transform time 
reading_time = time()
images = convert_from_path(FILE, grayscale=True, dpi=350)

print('Readig file: ...' + str(FILE[-12:] + '.Time: ' + str(time()-reading_time)) + ' secs')

# try to load the multi-threads based on eacch cpu
try:
    Threads= cpu_count()
except:
    Threads = 1

def main_ocr(image):
    image = array(image)
    image[image<=125]=0
    image[image>126] = 255
    
    #appy image enhanced if necessary (typically a blur works fine)
    image = blur(image, (2,2))
    return image_to_data(image, output_type=Output.DICT)

#%
ocr_time = time()
# apply ocr in dictionary format #pre-proc the text applying tokenizer and  stem if needed 
dictionaries = []
with ThreadPoolExecutor(max_workers=Threads-1) as executor:  # load as many threads availabe but one for general porpuses
   results = executor.map(main_ocr,images)     # return the process in order 
   [dictionaries.append(result) for result in results]

print('Applying OCR in ' + str(len(images)) + ' pages. Time: ' + str(time()- ocr_time) + '. Threads: ' + str(Threads-1))




#%                        SET THE CONFIGURATIONS

ner_time = time() #time NLP model 

try:
    nlp('model?')
except:
    nlp = load(MODEL_PATH)

print('Loading NLP Model. Time: ' + str(time()-ner_time))

# load the configuration files and declare grammar rules 

TOPICS = ConfigObj('./config/config_topic.ino')     # load the interest points
ENTS = ConfigObj('./config/config_ents.ino')      # load the NAME ENTITY RULES 
#open_exp = r"\b[A-Z][A-Z]+\b"                               # reg exp rules                
open_exp = r"[A-Za-z]\w+"
mon_exp = r"[^0-9\.0-9]+"
alpha_exp = r"(?=[A-Za-z])\w+.*(?=[a-z])\w+."

page = 0
suspects = lossrun.search_rules(dictionaries[page], TOPICS)  # store the suspects (topic fit)
spatial_filter_ver, tops, spatial_filter_hor, lefts = lossrun.spatial_filter(dictionaries[page], suspects, 'LOSSRUN') 

claims = []

img = array(images[page])

claims_time = time()
# extract the claims in the report
claims_policies = []
for idx, word in enumerate(dictionaries[page]['text']):
    if lossrun.isaclaim(word):
        x_1 = dictionaries[page]['top'][idx]
        y_1 = dictionaries[page]['left'][idx]
        delta = dictionaries[page]['width'][idx]
        chars, nums, punks = lossrun.count_chars(word)
        claims_policies.append([word, x_1, y_1, delta, chars, nums, punks]) 
claims_policies
claim_b =[]
policy_b = []
model = lossrun.get_NB()
for claim in claims_policies:
    if model.predict(array([claim[1:]])) == 0: # 0 == policy
        policy_b.append(claim[0])
    elif model.predict(array([claim[1:]])) == 1: # 1 == claim, -1 == other sus 
        claim_b.append(claim[0])
    else:
        claim_b.append(claim[0])

#print('claims', f'{claim_b}')
#print('policies', f'{policy_b}')

# get the sections in reports for each claim
coords = []
for i, claim, in enumerate(dictionaries[page]['text']):
    if claim in claim_b:
        top = dictionaries[page]['top'][i]
        left  = dictionaries[page]['left'][i]
        width  = dictionaries[page]['width'][i]
        height  = dictionaries[page]['height'][i]
        coords.append(([top, left, width, height]))
        line(img,(left,top),(max(dictionaries[page]['left']), top),(0,0,0), 10)
    if claim in policy_b:
        top = dictionaries[page]['top'][i]
        left  = dictionaries[page]['left'][i]
        width  = dictionaries[page]['width'][i]
        height  = dictionaries[page]['height'][i]
        #coords.append(([top, left, width, height]))
        line(img,(left,top),(max(dictionaries[page]['left']), top),(0,0,0), 25)

coords.append(([max(dictionaries[page]['top']), 0, 0, 0]))
print('Processing claims... Time' + str(time() - claims_time))
# %
#figure(figsize=(17,15))
#imshow(img, cmap='gray')

#%%                          EXTRACT THE INFO 
info_time = time()
stus = []
results = []
for i, ENT in enumerate(suspects):
    sent_hor  = nlp(' '.join(spatial_filter_hor[i]))
    sent_ver =  nlp(' '.join(spatial_filter_ver[i]))

    if ENTS[ENT[0]] == []:
        aux = re.findall(open_exp, ' '.join(spatial_filter_hor[i]))
        sentence = ' '.join(aux)
        results.append([sentence, suspects[i][-1], suspects[i][0]])
    
    elif ENT[0] == 'status':
        [results.append([word, tops[i][k],  'status']) for k, word in enumerate(spatial_filter_ver[i]) if 'CL' in word.upper()]
        [results.append([word, tops[i][k],  'status']) for k, word in enumerate(spatial_filter_ver[i]) if 'OP' in word.upper()]
    elif '-' in ENTS[ENT[0]]:
        pass

    for ent in sent_ver.ents:
        if ent.label_ in ENTS[ENT[0]]:
            if ent.text in spatial_filter_ver[i]:
                word_top = tops[i][spatial_filter_ver[i].index(ent.text)]
                results.append([ent.text, word_top, suspects[i][0]])
  
    [results.append([ent.text, suspects[i][-1],  suspects[i][0]]) for ent in sent_hor.ents if ent.label_ in ENTS[ENT[0]]]
    #[results.append([ent.text, tops[i][-1],  suspects[i][0]]) for ent in sent_ver.ents if ent.label_ in ENTS[ENT[0]]]

print('Info extracted... Time:' + str(time() - info_time))



#%%                                GROUP THE INFO IN EVERY CLAIM                                   
results
outputs = defaultdict(list) 

for i, result in enumerate(results):
    if result[0] != '':
        x1 = result[1] + coords[0][1]//2
        if x1 < coords[0][0]:
            outputs['cover'].append(result[2] + ': ' + result[0])
        else:
            for j in range(len(coords)-1):
                if coords[j][0] < x1 < coords[j + 1][0]:
                    outputs[claim_b[j]].append(result[2] + ': ' + result[0])

outputs

#%%

coords
#%%
imshow(array(images[0]))

#%%
suspects
