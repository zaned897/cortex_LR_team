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
import json
import lossrun

from sys import path
from os import getcwd
from numpy import array
from pandas import DataFrame
from pandas import isnull
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
from datetime import datetime



def run (FILE = 'data/lossruns/SEDGWICK.pdf'):
    
    path.append(getcwd())
    print("\tLossrun Reports Info Extraction\nDemo version by Asymm Developers & and Now Insurance\ndate:" + str(datetime.now()))
    print('.'*50 + '\n'*3)


    MODEL_PATH = 'models/lossrun_models/latest' # NLP model, contains the NER PARSER AND TAGGER
    OUTPUT_FILE = True
    #---------------------------------------------------------------------------------------
    #%                         LOAD DATA
    #---------------------------------------------------------------------------------------

    # default files

    # time the pdf to image transform time 
    reading_time = time()
    images = convert_from_path(FILE, grayscale=True, dpi=350)
    print('Readig file: ...' + str(FILE[-12:] + '.Time: ' + str(time()-reading_time)) + ' secs')

    # try to load the multi-threads based on each cpu
    try:
        Threads= cpu_count()
    except:
        Threads = 1

    # Natural language processign for multi-threads
    # NLP model must be loaded 
    def multi_nlp(string):
        sent = nlp(string)
        return sent # return the result in dictionary output

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
    #                        SET THE CONFIGURATIONS
    #---------------------------------------------------------------------------------------


    # time the nlp analysis
    ner_time = time()

    try:
        nlp('model?') # if model already loaded, don't load it again
    except:
        nlp = load(MODEL_PATH)

    print('Loading NLP Model. Time: ' + str(time()-ner_time))

    # load the configuration files and declare the grammar rules 
    TOPICS = ConfigObj('LOSSRUNS/config/config_topic.ino')     # load the interest points
    ENTS = ConfigObj('LOSSRUNS/config/config_ents.ino')      # load the NAME ENTITY RULES 
    #open_exp = r"\b[A-Z][A-Z]+\b"                               # reg exp rules                
    open_exp = r"[A-Za-z]\w+" # for open text typically based on mayus in the report desciption
    mon_exp = r"(\$)(.*)(\d)|(\d+)(.*)(\.\d+)" # for money format (not used if NLP performs)
    alpha_exp = r"(?=[A-Za-z])\w+.*(?=[a-z])\w+." # for alpha/num (licences) (not used if NLP performs)
    date_exp = r"(\d+)(/)(\d+)(/)(\d+)"


    #---------------------------------------------------------------------------------------
    #%                        EXTRACT THE INFO 
    #---------------------------------------------------------------------------------------

    outputs = defaultdict(list) 
    results = []
    for dictionary in dictionaries:
        
        # store the topics in the report that fit with synonims (can apply stem or edit distance)
        # suspects = lossrun.search_rules(dictionary, TOPICS) 
        suspects = lossrun.searchTopics(dictionary, TOPICS) 

        suspects_indices = []
        [suspects_indices.append(suspect[2]) for suspect in suspects]
        # extract the text in the same raw and column of each topic
        #spatial_filter_ver, tops, spatial_filter_hor, lefts = lossrun.spatial_filter(dictionary, suspects, 'LOSSRUN') 
        spatial_filter_ver, tops, spatial_filter_hor, lefts  = lossrun.searchSameRowCol(dictionary, suspects_indices) 

        # time the claims search 
        claims_time = time()

        # extract the text in report that fit the rules of policies and claims
        
        claims, policies = lossrun.getClaimsPolicies(dictionary)

        if len(claims) == 0:
            claims, policies = lossrun.getClaimsPolicies(dictionary, length = 6)
        print(claims)
        print(policies)
        
    
        img = array(images[page])
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
            print('!' + str(len(claims)) + ' claims were founded...\n')
        else:
            print('NO CLAIMS FOUND!')
        """ uncoment this if want to plot the claims in reports
        figure(figsize=(17,15))
        imshow(img, cmap='gray')
        """

        stus = []
        results = []
        data_poly = json.load(open('LOSSRUNS/config/poly_data.json'))
        #%
        # extract normal cases
        for i, ENT in enumerate(suspects):
            
            found_ent = False
            # get the text in files and the colums for each suspects
        
            output_type = ENTS[ENT[0]][0]
            # open text output expected
            if output_type == 'OPENTXT':
                aux = re.findall(open_exp, ' '.join(spatial_filter_hor[i]))
                sentence = ' '.join(aux)
                results.append([sentence, suspects[i][-1], ENT[0]])
            
            
            # Binary output expected
            elif output_type == 'POLY':
                for j, word in enumerate(spatial_filter_ver[i]):
                    for k in range(len(data_poly[ENT[0]])):
                        if data_poly[ENT[0]][k]['abbreviation'].upper() == word.upper() or data_poly[ENT[0]][k]['name'].upper() == word.upper():
                            results.append([data_poly[ENT[0]][k]['name'], tops[i][j], ENT[0]])
                            
            elif output_type == 'ENT':

                # horizontal entities
                sent_hor = nlp(' '.join(spatial_filter_hor[i]))
                
                for ent in sent_hor.ents:
                    if ent.label_ in ENTS[ENT[0]][1:]:
                        if ent.label_ == 'MONEY':
                            pass
                        
                        else:
                            results.append([ent.text, suspects[i][-1],  ENT[0]])
                            found_ent = True
                #[results.append([ent.text, suspects[i][-1],  ENT[0]]) for ent in sent_hor.ents if ent.label_ in ENTS[ENT[0]]]

                # vertical entities
                if not found_ent:
                    
                    if 'DATE' in ENTS[ENT[0]]:
                        for l, elmn in enumerate(spatial_filter_ver[i]):
                            try:
                                results.append([''.join(re.findall(date_exp, elmn)[0]), tops[i][l], ENT[0]])
                            except:
                                pass
                    elif 'MONEY' in ENTS[ENT[0]]:
                        for l, elmn in enumerate(spatial_filter_ver[i]):
                            try:
                                results.append([''.join(re.findall(mon_exp, elmn)[0]),tops[i][l], ENT[0]])
                            except:
                                pass
                    
                    else:
                        for l, elmn in enumerate(spatial_filter_ver[i]):    
                            spatial_filter_ver[i][l] = ENT[0] + ': ' + elmn

                        sents_ver = []
                        with ThreadPoolExecutor(max_workers= Threads - 1)  as executor:  # load as many threads availabe but one for general porpuses
                            nlp_results = executor.map(multi_nlp, spatial_filter_ver[i])     # return the results of the ocr in sorted way 
                            [sents_ver.append(result) for result in nlp_results]
                
                        for k, sent in enumerate(sents_ver):
                            for ent in sent.ents:
                                if ent.label_ in ENTS[ENT[0]][1:]:
                                    results.append([ent.text, tops[i][k], ENT[0]])
                                
            
        topics_founded = [result[-1] for result in results]
        res_cros = []
        for topic in TOPICS:
            if '-' in TOPICS[topic][0] and topic not in topics_founded:
                topic_compose = list(TOPICS[topic][0].split('-'))
                res_cros.append(lossrun.interWordBased(dictionary, topic_compose, topic))
                
        for res in res_cros:
            for sub in res:
                results.append(sub)


        #---------------------------------------------------------------------------------------
        #%                                GROUP THE INFO IN EVERY CLAIM                                   
        #---------------------------------------------------------------------------------------

        # time the time for gropuing the results
        grouping_time = time()
        outputs['cover']
        for i, result in enumerate(results):
            if result[0] != '':
                x1 = result[1] + coords[0][-1] // 2
                if x1 < coords[0][0]:
                    outputs['cover'].append(result[2] + ': ' + result[0])
                else:
                    for j in range(len(coords)-1):
                        if coords[j][0] < x1 < coords[j + 1][0]:
                            outputs[claims[j]].append(result[2] + ': ' + result[0])
        print('Results sorting. Time: ' + str(time()-grouping_time)+ 'secs \n')
        print('..'*50)


    if  not OUTPUT_FILE:
        return 0
    #------------------------------------------------------------
    #           EXPORT THE RESULTS
    #------------------------------------------------------------

    # It is neecesary move thos to main library
    def try_parsing_date(text):
        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%d/%m/%y', '%m/%d/%y', '%m/%d/%Y'):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        raise ValueError('no valid date format found')

    cover_topics = ['insured', 'as_of']
    topics_topic = [topic for topic in TOPICS]
    frame = DataFrame(columns = topics_topic, index = range(len(outputs)))


    for idx, out in enumerate(outputs):
        for res in outputs[out]:
            topico = res.split(':')[0]
            resultado = res.split(':')[1]
            if isnull(frame.loc[idx, topico]):
                frame.xs(idx)[topico] = resultado
            if any('status: open' in one.lower() for one in outputs[out]) and topico == 'exp_date':
                frame.xs(idx)[topico] = '--'
            elif not isnull(frame.loc[idx, topico]) and (topico == 'report_date' or topico == 'exp_date') and  try_parsing_date(resultado.replace(' ','')) >  try_parsing_date(frame.loc[idx, topico].replace(' ','')):
                frame.xs(idx)[topico] = resultado
            frame.xs(idx)['claim'] = out
            
            # add data extracted from the coverage page 

        if isnull((frame.loc[idx, 'as_of'])):
            frame.xs(idx)['as_of']  = frame.loc[list(frame['claim']).index('cover'), 'as_of']
        
        if isnull((frame.loc[idx, 'insured'])):
            frame.xs(idx)['insured']  = frame.loc[list(frame['claim']).index('cover'), 'insured']
        

    frame =  frame.dropna(how = 'all', axis = 1)
    frame.to_csv('results/' + FILE.split('/')[-1][:-4] + ' results.csv')

    return frame.to_dict('records')

run()    #EOC