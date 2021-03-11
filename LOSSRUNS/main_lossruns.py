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

from pandas.io.parsers import FixedWidthReader
import lossrun

from numpy import array
from spacy import load
from matplotlib.pyplot import cla, figure
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


#---------------------------------------------------------------------------------------
#%                         LOAD DATA
#---------------------------------------------------------------------------------------

# default files
FILE = '../data/lossruns/MultipleClaims4.pdf' # file pdf to process

MODEL_PATH = '../models/lossrun_models/lr_lt_v1' # NLP model, contains the NER PARSER AND TAGGER

# time the pdf to image transform time 
reading_time = time()
images = convert_from_path(FILE, grayscale=True, dpi=350)
print('Readig file: ...' + str(FILE[-12:] + '.Time: ' + str(time()-reading_time)) + ' secs')

# try to load the multi-threads based on each cpu
try:
    Threads= cpu_count()
except:
    Threads = 1


# Optical Character Recognition function
def main_ocr(image):
    image = array(image)  # transform PIL image to array to binarize
    image[image <= 125] = 0
    image[image > 126] = 255
    image = blur(image, (2,2)) # apply a blur whit kernel 2 x 2
    return image_to_data(image, output_type=Output.DICT) # return the result in dictionary output

# time the  ocr process
ocr_time = time()

# apply ocr and store the dictionaries in a list of results
dictionaries = []
with ThreadPoolExecutor(max_workers=Threads-1) as executor:  # load as many threads availabe but one for general porpuses
   ocr_results = executor.map(main_ocr,images)     # return the results of the ocr in sorted way 
   [dictionaries.append(result) for result in ocr_results]

print('Applying OCR in ' + str(len(images)) + ' pages. Time: ' + str(time()- ocr_time) + '. Threads: ' + str(Threads-1))


# join all pages in report is a single dictinoary

# create a dictionary base
dictionary = {'level':[],'page_num':[],'block_num':[],'par_num':[],'line_num':[],'word_num':[],'left':[],'top':[],'width':[],'height':[],'conf':[],'text':[]}

# concatenate the dictinoary augmenting the size in y axis 
for page, dix in enumerate(dictionaries):
    dix['top'] = list(array(dix['top']) + array(images[page]).shape[0] * page)

    # add the items in each page to a gral dictionary, y axis is increased by 
    # page size
    for item in dix: 
        dictionary[item] = dictionary[item] + dix[item]

#---------------------------------------------------------------------------------------
#%                        SET THE CONFIGURATIONS
#---------------------------------------------------------------------------------------


# time the nlp analysis
ner_time = time()

try:
    nlp('model?') # if model already loaded, don't load it again
except:
    nlp = load(MODEL_PATH)

print('Loading NLP Model. Time: ' + str(time()-ner_time))


# load the configuration files and declare the grammar rules 
TOPICS = ConfigObj('./config/config_topic.ino')     # load the interest points
ENTS = ConfigObj('./config/config_ents.ino')      # load the NAME ENTITY RULES 
#open_exp = r"\b[A-Z][A-Z]+\b"                               # reg exp rules                
open_exp = r"[A-Za-z]\w+" # for open text typically based on mayus in the report desciption
mon_exp = r"[^0-9\.0-9]+" # for money format (not used if NLP performs)
alpha_exp = r"(?=[A-Za-z])\w+.*(?=[a-z])\w+." # for alpha/num (licences) (not used if NLP performs)





#---------------------------------------------------------------------------------------
#%                          APPLY NATURAL SEARCHING ALGORITHM
#---------------------------------------------------------------------------------------

#page = 0 # in order to process the hole report change the code  to process each page or merge the dictionaries

# store the topics in the report that fit with synonims (can apply stem or edit distance)
suspects = lossrun.search_rules(dictionary, TOPICS) 
# extract the text in the same raw and column of each topic
spatial_filter_ver, tops, spatial_filter_hor, lefts = lossrun.spatial_filter(dictionary, suspects, 'LOSSRUN') 

# time the claims search 
claims_time = time()

# extract the text in report that fit the rules of policies and claims
claims_policies = []
img = array(images[page])
for idx, word in enumerate(dictionary['text']):
    if lossrun.isaclaim(word):
        features = lossrun.get_features(dictionary, word, idx)
       # x_1 = dictionary['top'][idx]
       # y_1 = dictionary['left'][idx]
       # delta = dictionary['width'][idx]
       # features = lossrun.count_chars(word)
        claims_policies.append(features) 



#---------------------------------------------------------------------------------------
#%                               FIND THE CLAIMS IN THE REPORTS
#---------------------------------------------------------------------------------------

#%
# get the trained model Naive Bayes model
data_train_nb = '../test/processed_NB_train_data.csv'
train_percent = .8
model = lossrun.get_NB(data_train_nb, train_percent)

# extract the claims and policies in report according NB model
claims = []
policies = []
extra = []
for claim in claims_policies:
    if model.predict(array([claim[1:]]) + 1) == 0: # 0 == policy
        policies.append(claim[0])
    elif model.predict(array([claim[1:]]) + 1) == 1: # 1 == claim, -1 == other sus 
        claims.append(claim[0])
    else:
        extra.append(claim)

print('claims', f'{claims}')
print('policies', f'{policies}')

# get the sections in reports for each claim
coords = []
for i, claim, in enumerate(dictionary['text']):
    if claim in claims:
        top = dictionary['top'][i]
        left  = dictionary['left'][i]
        width  = dictionary['width'][i]
        height  = dictionary['height'][i]
        coords.append(([top, left, width, height]))
        line(img,(left,top),(max(dictionary['left']), top),(0,0,0), 10)
    if claim in policies:
        top = dictionary['top'][i]
        left  = dictionary['left'][i]
        width  = dictionary['width'][i]
        height  = dictionary['height'][i]
        #coords.append(([top, left, width, height]))
        line(img,(left,top),(max(dictionary['left']), top),(0,0,0), 25)

coords.append(([max(dictionary['top']), 0, 0, 0]))
print('Processing claims... Time: ' + str(time() - claims_time))

if not lossrun.there_are_claims(dictionary, claims):
    print('THERE ARE ' + str(len(claims)) + ' CLAIMS FOUNDED')
else:
    print('NO CLAIMS FOUND!')
""" uncoment this if want to plot the claims in reports
figure(figsize=(17,15))
imshow(img, cmap='gray')
"""



#---------------------------------------------------------------------------------------
#%                          EXTRACT THE INFO 
#---------------------------------------------------------------------------------------
#%
# time the results
info_time = time()
stus = []
results = []


# extract normal cases
for i, ENT in enumerate(suspects):

    # get the text in files and the colums for each suspects
    sent_hor  = nlp(' '.join(spatial_filter_hor[i]))
    sent_ver =  nlp(' '.join(spatial_filter_ver[i]))

    output_type = ENTS[ENT[0]][0]
    # open text output expected
    if output_type == 'OPENTXT':
        aux = re.findall(open_exp, ' '.join(spatial_filter_hor[i]))
        sentence = ' '.join(aux)
        results.append([sentence, suspects[i][-1], suspects[i][0]])
    
    
    # Binary output expected
    elif output_type == 'BIN':
        [results.append([word, tops[i][k],  'status']) for k, word in enumerate(spatial_filter_ver[i]) if 'CL' in word.upper()]
        [results.append([word, tops[i][k],  'status']) for k, word in enumerate(spatial_filter_ver[i]) if 'OP' in word.upper()]


    elif output_type == 'ENT':
        # Entity output expected 
        for ent in sent_ver.ents:
            if ent.label_ in ENTS[ENT[0]]:
                if ent.text in spatial_filter_ver[i]:
                    word_top = tops[i][spatial_filter_ver[i].index(ent.text)]
                    results.append([ent.text, word_top, suspects[i][0]])
        
        [results.append([ent.text, suspects[i][-1],  suspects[i][0]]) for ent in sent_hor.ents if ent.label_ in ENTS[ENT[0]]]
        #[results.append([ent.text, tops[i][-1],  suspects[i][0]]) for ent in sent_ver.ents if ent.label_ in ENTS[ENT[0]]]
 
# extract cross cases

res_cros = []
for topic in TOPICS:
    if '-' in TOPICS[topic][0]:
        topic_compose = list(TOPICS[topic][0].split('-'))
        res_cros.append(lossrun.interWordBased(dictionary, topic_compose, topic))
        
for res in res_cros:
    for sub in res:
        results.append(sub)

#    for topic in TOPICS:
#        if '-' in TOPICS[topic][0]:
#            print(topic)
#%
print('Info extracted... Time:' + str(time() - info_time))



#---------------------------------------------------------------------------------------
#%                                GROUP THE INFO IN EVERY CLAIM                                   
#---------------------------------------------------------------------------------------

# time the time for gropuing the results
grouping_time = time()

outputs = defaultdict(list) 

for i, result in enumerate(results):
    if result[0] != '':
        x1 = result[1] + coords[0][1]//2
        if x1 < coords[0][0]:
            outputs['cover'].append(result[2] + ': ' + result[0])
        else:
            for j in range(len(coords)-1):
                if coords[j][0] < x1 < coords[j + 1][0]:
                    outputs[claims[j]].append(result[2] + ': ' + result[0])
print('Results sorting. Time: ' + str(time()-grouping_time)+ 'secs')


print('The results are:')
for output in outputs:
    print('.'*50)
    print(output, outputs[output])

outputs

#EOC
#%%
results
