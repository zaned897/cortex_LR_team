# %%                            LOAD DEPENDENCIES
from pdf2image import convert_from_path
import pytesseract as pt 
from pytesseract import Output
import time
from PIL import Image
import concurrent.futures
import lossrun
import lossrun_models
from configobj import ConfigObj
import spacy
import re 
import numpy as np
import sys
from datetime import datetime
from skimage.transform import rescale
from skimage.transform import resize
from cv2 import blur
from cv2 import imread
from cv2 import IMREAD_GRAYSCALE
from skimage.metrics import structural_similarity
from skimage.filters import unsharp_mask
import sys
import multiprocessing

##                                    DATA READ AND PROCESS
#_________________________________
PATH = 'data/NPDBQA3.pdf'  
#_________________________________

print('Reading the reports and remove non info pages from report %s...' % (PATH))
# time it
reading_time = time.time()
# Transform entire pdf format to text
new_shape = (200,200)
images_non_filter = convert_from_path(PATH, grayscale=True) # conversion
print('reading image time: %f' % (time.time()-reading_time))

filter_time = time.time()
filter_no_threads = time.time()
list_of_relevant = lossrun.non_info_filter(PATH, score = 0.55)
images_filter = []
[images_filter.append(image) for i, image in enumerate(images_non_filter) if list_of_relevant[i]]

[print('pages removed: %d' % (i + 1)) for i, image in enumerate(images_non_filter) if not list_of_relevant[i]]
print('filtering time: %f' % (time.time() - filter_time))


#  OCR AND CLAIMS EXTRACTION
# extract txt in image
print('Applying OCR and extract each claim in report...')
get_claims_time = time.time()
def extract_text(image): 
    # rescale image if necesary
    #image_255 = unsharp_mask(np.array(image),radius=5)*255
    image_255 = blur(np.array(image),(2,2))
    return pt.image_to_data(image_255,output_type=Output.DICT)
# save a dict for each page 
dictionaries = []
# process each image width multi-threads
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor: 
    results = [executor.submit(extract_text, image) for image in images_filter]
    for result in concurrent.futures.as_completed(results):
        dictionaries.append(result.result())
all_dcn_list = []
# get all the DCN number in NPDB files
for _dict in dictionaries:
    if 'DCN:'in _dict['text']:
        all_dcn_list.append(_dict['text'][_dict['text'].index('DCN:') + 1])
# remove repeated values
dcn_list = list(dict.fromkeys(all_dcn_list))
claims = []
# haga busqueda e incerción
for dcn in dcn_list:
    aux = {'level':[],'page_num':[],'block_num':[],'par_num':[],'line_num':[],'word_num':[],'left':[],'top':[],'width':[],'height':[],'conf':[],'text':[]}
    i = 0
    for _dict in dictionaries:
        if dcn in _dict['text']:        
            if _dict['top'][_dict['text'].index(dcn)] < 100:
                _dict['top']  = list(np.array(_dict['top']) + (2200 * i))
                i += 1
                for key in _dict.keys():
                    aux[key] = aux[key] + _dict[key]
    
    
    claims.append(aux)
print('OCR processed in %f' % (time.time() - get_claims_time))



# %                 SEARCHING FOR TOPICS IN CLAIMS
print('Processing Topics of the CONFIGURATION TOPICS file')
inject_time = time.time()
try:
    nlp('model?')
except:
    print('loading NPDB NER model...')
    nlp = spacy.load('/home/zned897/Proyects/pdf_text_extractor/NPDB_ner_model')

topic_conf = ConfigObj('config/config_npdb_topics.ino')
Topics = ConfigObj('./config/config_npdb_entites.ino')
open_exp = r"\b[A-Z][A-Z]+\b"
mon_exp = r"[^0-9\.0-9]+"
alpha_exp = r"(?=[A-Za-z])\w+.*(?=[a-z])\w+."

# search for topics and entities
for dcn, claim in enumerate(claims):
    suspects = lossrun.search_rules(claim, topic_conf)
    spatial_filter = lossrun.spatial_filter(claim, suspects, 'NPDB')
    spatial_filter_topics = len(spatial_filter)

    #extract data    
    pract_name, ref_number, ent_name, paid_by, outcome, init_act , act_basis = [],[],[],[],[],[],[]
    # get the entities for each sptatial relation
    
    total_amount = ''
    amount = ''
    npi = ''
    licensure = []
    proc_date = [datetime(1900,1,1)]
    event_date = [datetime(1900,1,1)]
    paid_date = [datetime(1900,1,1)]
    relevant = True

    for topic in range(spatial_filter_topics):
        sentence  = ' '.join(spatial_filter[topic])
        sentence = re.sub(r"\s+",' ', sentence)
        doc = nlp(sentence)
        for ent in doc.ents:
            if suspects[topic][0] == 'pract_name':
                if ent.label_ in Topics['pract_name']:
                    pract_name.append(ent.text)
            if suspects[topic][0] == 'ref_number':
                if ent.label_ in Topics['ref_number']:
                    ref_number.append(ent.text)
            if suspects[topic][0] == 'proc_date':
                if ent.label_ in Topics['proc_date']:
                    try:
                        proc_date.append(datetime.strptime(ent.text, '%m/%d/%Y'))
                    except:
                        pass
            if suspects[topic][0] == 'paid_date':
                if ent.label_ in Topics['paid_date']:
                    try:
                        paid_date.append(datetime.strptime(ent.text, '%m/%d/%Y'))
                    except:
                        pass
            if suspects[topic][0] == 'event_date':
                if ent.label_ in Topics['event_date']:
                    try:
                        event_date.append(datetime.strptime(ent.text, '%m/%d/%Y'))
                    except:
                        pass
            if suspects[topic][0] == 'ent_name':
                if ent.label_ in Topics['ent_name']:
                    ent_name.append(ent.text)
            if suspects[topic][0] == 'paid_by':
                if ent.label_ in Topics['ent_name']:
                    paid_by.append(ent.text)
            if suspects[topic][0] == 'total_amount':
                if ent.label_ in Topics['total_amount']:
                    total_amount = ent.text
            if suspects[topic][0] == 'amount':
                if ent.label_ in Topics['amount']:
                    amount = ent.text
            if suspects[topic][0] == 'npi':
                if ent.label_ in Topics['npi']:
                    npi = ent.text
       #     if suspects[topic][0] == 'licence':  
        #        if ent.label_ in Topics['licence']:
         #           licensure = ent.text
        if suspects[topic][0] == 'outcome':
            outcome.append(sentence)
        if suspects[topic][0] == 'init_act':   
            aux = re.findall(open_exp, sentence)
            sentence = ' '.join(aux)
            init_act.append(sentence) if sentence is not '' else None
        if suspects[topic][0] == 'act_basis': 
            aux = re.findall(open_exp, sentence)
            sentence = ' '.join(aux)
            act_basis.append(sentence) if sentence is not '' else None
        if suspects[topic][0] == 'relevant':
            relevant = False
        if suspects[topic][0] == 'licence':
           licensure.append(str(re.sub(alpha_exp,'', sentence)))

    pract_name = str(pract_name)[1:-1]
    ref_number = str(ref_number)[1:-1]
    ent_name =  str(ent_name)[1:-1]
    paid_by = str(paid_by)[1:-1]
    total_amount = int(float(re.sub(mon_exp, '','0' + total_amount)))
    amount = int(float(re.sub(mon_exp, '','0' + amount)))
    outcome = str(outcome)[1:-1]
    init_act = str(init_act)[1:-1]
    act_basis = str(act_basis)[1:-1]
    licensure = str(licensure)[2:-1]
    try: 
        event_date = [min(event_date[1:])]
    except:
        event_date = [datetime(1900,1,1)]

    print('.'*50)
    print('pract_name:' + pract_name)
    print('proc_date:' + str(max(proc_date)))
    print('paid_date:' + str(max(paid_date)))
    print('event_date:' + str(event_date))
    print('total_amount:' + str(total_amount))
    print('amount:' + str(amount))
    print('outcome:' + str(outcome))
    print('ent_name:' + str(ent_name))
    print('init_act:' + str(init_act))  
    print('act_basis:' + str(act_basis))
    print('NPI:' + npi)
    print('Licencure:' + licensure)
    print('DCN:' + dcn_list[dcn])
    
    
    lossrun_models.npdbRecord(process_date = max(proc_date),
                          practitioner_name = pract_name,
                          action_initial = init_act,                  
                          action_basis = act_basis, 
                          entity_name = ent_name,
                          payment_date = max(paid_date), 
                          payment_total_amount = total_amount,
                          event_day = min(event_date), 
                          event_outcome = outcome,
                          event_paid_by = licensure,
                          npi = npi, 
                          amount_pract = amount,
                          dcn = dcn_list[dcn]
                          )
    
    
print('Ingection time: %f' %(time.time()- inject_time ))

# END OF CODE