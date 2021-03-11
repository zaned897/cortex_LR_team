# %%                           LOAD DEPENDENCIES
from pdf2image import convert_from_path # utils from poppler, .pdf -> .png
import pytesseract as pt                # ocr tool from google eng dict enable 
from pytesseract import Output          # ocr process stored in dictionary format
import time                             # check process times
from PIL import Image                   # imaging library
import concurrent.futures               # multi-threads
from configobj import ConfigObj         # handle the configuration files
import spacy                            # NLP library
import re                               # regular expression for open text  
import numpy as np                      # math library
import sys, getopt, multiprocessing                    # system utils and get options
from datetime import datetime           # date time
from skimage.transform import rescale   # image proccessing
from skimage.metrics import structural_similarity
from skimage.filters import unsharp_mask
from cv2 import blur, imread,  IMREAD_GRAYSCALE     # computer vision
import lossrun                          # local tools for loss run reports



def run(PATH='/home/app/NPDB/data/NPDBQA6.pdf'):
    #%%                         DEFINE THE OPTIONS OF THE CODE
    args_list = sys.argv[1:]    # options: file to process, path to nlp model and database available?
    args_flags = 'hf:m:d:'      # short options
    args_long = ['help', 'file-path=','model-path=', 'data-base-insert='] # long format for options
    argv = sys.argv[1:]         # ignore the first arg since is the script.py opt
    data_results = []

    # catch the rigth options
    try:
       argument, values = getopt.getopt(args_list, args_flags, args_long)
    except getopt.error as err: 
        print(str(err) + ', system exit...')
        sys.exit(2)            # there are an error with the input then exit


    # Set the default nlp model and database insertion 
    MODEL_PATH = '/home/app/models/npdb/latest/'
    db = False

    # assign the inputs to the script variables
    for current_argument, current_value in argument:
        if current_argument in ['-h','--help']:
            print(lossrun.print_help())
        elif current_argument in ['-f', '--file-path']:
            PATH = str(current_value)
            print('Processing file: '+ str(current_value))
        elif current_argument in ['-m', '--model-path']:
            MODEL_PATH = str(current_value)
            print('Model directory: ' + str(current_value))
        elif current_argument in ['-d', '--data-base-insert']:
            db = [False, True][current_value.lower()[0] == 't' or current_value.lower()[0]=='y']
            print('Database insetion: ', db)


       


    #%%                                  FOR THE DEBUGGING UNCOMMENT THE FOLLOW BLOCK
    # manual input 
    """
    PATH = './data/NPDBQA23.pdf'
    MODEL_PATH = '/home/zned897/Proyects/pdf_text_extractor/NPDB_ner_model'
    db = False
    """




    #%%                                  DATA READ AND PROCESS

    print('Reading the reports and remove non info pages from report %s...' % (PATH))
    # time it
    reading_time = time.time()
    # Transform entire pdf format to text
    new_shape = (200,200)   # compress the image to check the image similarity faster
    images_non_filter = convert_from_path(PATH, grayscale=True) # conversion pdf -> image
    print('reading image time: %f' % (time.time()-reading_time))# time it

    # get cover pages info
    # pull as many pages as threads in cpu
    try:
        THREADS = multiprocessing.cpu_count()
    except:
        THREADS = 1
        print("Cannot identify the Threads number, Threads = 1")

    # OCR main function
    def extract_text(image): 
        # rescale image if necesary
        # image_255 = unsharp_mask(np.array(image),radius=5)*255
        image_255 = blur(np.array(image),(2,2)) # blur variates in every cpu, check a different image proc
        return pt.image_to_data(image_255,output_type=Output.DICT)

    # check the cover pages info
    print("Processing cover pages ...")
    cover_dictionaries = []
    # process each image width multi-threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS//2) as executor:  # add threads as max worker when sagemaker up
       results = executor.map(extract_text,images_non_filter[:5])     # return the process in order 
       [cover_dictionaries.append(result) for result in results]




    #%%                         PROCESS COVER PAGES INFO
    # search if the final sentense of cover pages are present and npi, fein, dea numbers found
    END_OF_COVER_PAGES = ['NO REPORTS FOUND','UNABRIDGED REPORT(S) FOLLOW']
    NPI_FOUND = False
    FEIN_FOUND = False
    DEA_FOUND = False
    DCN_COVER = False
    COVER_PAGE_FOUND=False
    all_dcn_list = []

    #chek if there is dcn for the cover pages and extract 
    for page, _dict in enumerate(cover_dictionaries):
        # search for the keywords: NPI, DEA, etc in the cover pages
        if ('NPI:' in ' '.join(_dict['text']).upper()) and not NPI_FOUND:   
            NPI = _dict['text'][_dict['text'].index('NPI:') + 1]
            NPI_FOUND = True # if it's found store it since it doesnt changes
        if ('FEIN:' in ' '.join(_dict['text']).upper()) and not FEIN_FOUND:   
            FEIN = _dict['text'][_dict['text'].index('FEIN:') + 1]
            FEIN_FOUND = True
        if ('DEA:' in ' '.join(_dict['text']).upper()) and not DEA_FOUND:   
            DEA = _dict['text'][_dict['text'].index('DEA:') + 1]
            DEA_FOUND = True                  
        if ('DCN:' in ' '.join(_dict['text']).upper()):   
            try:
                if _dict['top'][_dict['text'].index('DCN:')] < 100:
                    all_dcn_list.append(_dict['text'][_dict['text'].index('DCN:') + 1])
                    DCN_COVER = True
            except:
                pass

        elif len(re.findall(r"[0-9]{16}", ' '.join(_dict['text'])))>0:
            all_dcn_list.append(re.findall(r"[0-9]{16}", ' '.join(_dict['text']))[0])
            DCN_COVER = True

        if any(condition in ' '.join(_dict['text']).upper() for condition in END_OF_COVER_PAGES):
            starting_page = page
            COVER_PAGE_FOUND = True
            break

    # set process flags
    if not NPI_FOUND:
        NPI = ''    
    if not FEIN_FOUND:
        FEIN = ''    
    if not DEA_FOUND:
        DEA = ''   
    if not DCN_COVER:
        all_dcn_list.append('')
    # if the condition is not founded process from page 0
    if not COVER_PAGE_FOUND:
        starting_page  = 0 
        print('Cover pages not found')
    else:
        print('Cover pages found from 1 to %d'%(starting_page+1))




    #%%                             FILTER PAGES WITH NON RELEVANT INFO                           
    # continue whit the rest of report
    filter_time = time.time() # time it 
    list_of_relevant = lossrun.non_info_filter(PATH, score = 0.56)
    images_filter = []
    [images_filter.append(image) for i, image in enumerate(images_non_filter) if list_of_relevant[i]]
    dictionaries = []        
    # returnt the process as it completes (not necesary sorted)
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS//2) as executor: 
        results = [executor.submit(extract_text, image) for image in images_filter]
        for result in concurrent.futures.as_completed(results):
            dictionaries.append(result.result())

    [print('pages removed: %d' % (i + 1)) for i, image in enumerate(images_non_filter) if not list_of_relevant[i]]
    print('filtering time: %f' % (time.time() - filter_time))




    #%%                              OCR AND CLAIMS EXTRACTION
    # extract txt in image
    print('Applying OCR and extract each claim in report...')
    get_claims_time = time.time()
    # create a dictionary for every claim, dict 0 is the cover pages info (summary)
    claims = []
    # auxiliar dict to merge each ocr data in page
    aux = {'level':[],'page_num':[],'block_num':[],'par_num':[],'line_num':[],'word_num':[],'left':[],'top':[],'width':[],'height':[],'conf':[],'text':[]}
    page = 0 # defase page

    # process each list adding an offset of the starndar page sizes
    # process the cover pages
    for _dict in cover_dictionaries[:starting_page+1]:
        _dict['top']  = list(np.array(_dict['top']) + (2200 * page))
        page += 1
        for key in _dict.keys():
            aux[key] = aux[key] + _dict[key]
    claims.append(aux)    

    # proces the rest of the report
    # get all the claims associated to the DCN number
    for _dict in dictionaries:
        if 'DCN:'in _dict['text']:
            all_dcn_list.append(_dict['text'][_dict['text'].index('DCN:') + 1])
    # remove repeated values
    dcn_list = list(dict.fromkeys(all_dcn_list))

    # 
    for dcn in dcn_list[1:]:
        # auxiliar dict to merge each ocr data in page
        aux = {'level':[],'page_num':[],'block_num':[],'par_num':[],'line_num':[],'word_num':[],'left':[],'top':[],'width':[],'height':[],'conf':[],'text':[]}
        i = 0
        # process each page for each claim (non HISTORY DISCLOURE pages are processed)
        for _dict in dictionaries:
            if dcn in _dict['text']:        
                if _dict['top'][_dict['text'].index(dcn)] < 100:
                    _dict['top']  = list(np.array(_dict['top']) + (2200 * i))
                    i += 1
                    for key in _dict.keys():
                        aux[key] = aux[key] + _dict[key]
        claims.append(aux)
    # time it 
    print('OCR processed in %f' % (time.time() - get_claims_time))




    #%%                    SEARCHING FOR TOPICS IN CLAIMS
    print('Processing Topics of the CONFIGURATION TOPICS file')
    inject_time = time.time()
    try:
        nlp('model?')
    except:
        print('loading NPDB NER model...')
        #print(MODEL_PATH)
        nlp = spacy.load(MODEL_PATH)

    # load the configuration files and declare grammar rules 
    topic_conf = ConfigObj('/home/app/NPDB/config/config_npdb_topics.ino')     # load the interest points
    Topics = ConfigObj('/home/app/NPDB/config/config_npdb_entites.ino')      # load the NAME ENTITY RULES 
    open_exp = r"\b[A-Z][A-Z]+\b"                               # reg exp rules                
    mon_exp = r"[^0-9\.0-9]+"
    alpha_exp = r"(?=[A-Za-z])\w+.*(?=[a-z])\w+."
    licensure_exp = r"[A-Z0-9]\w+, [A-Z]{2}" 
   
    # %%   
    # search for topics and entities
    for dcn, claim in enumerate(claims[1:]): # search for dcn number in every claim 
        
        suspects = lossrun.search_rules(claim, topic_conf)  # store the suspects (topic fit) 
        spatial_filter = lossrun.spatial_filter(claim, suspects, 'NPDB')    # process the spatial relation as an NPDB
        spatial_filter_topics = len(spatial_filter)

        # extract data for each topic
        pract_name, ref_number, ent_name, paid_by, outcome, init_act , act_basis = [],[],[],[],[],[],[]
        # get the entities for each sptatial relation
        
        # cast the type of the vars
        total_amount = ''
        amount = ''
        npi = ''
        gender= ''
        ssn = ''
        fein =''
        subsequent_action = []
        type_of_adverse_action = []
        reasons_for_act = []
        dea = ''
        licensure = []
        proc_date = [datetime(1900,1,1)]
        event_date = [datetime(1900,1,1)]
        paid_date = [datetime(1900,1,1)]
        date_of_birth = [datetime(1900,1,1)]
        
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
                if (suspects[topic][0] == 'npi') and not NPI_FOUND:
                    if ent.label_ in Topics['npi']:
                        npi = ent.text
                if suspects[topic][0] == 'date_of_birth':
                    if ent.label_ in Topics['date_of_birth']:
                        try:
                            date_of_birth.append(datetime.strptime(ent.text, '%m/%d/%Y'))
                        except:
                            pass
                if (suspects[topic][0] == 'fein') and not FEIN_FOUND:
                    if ent.label_ in Topics['fein']:
                        fein = ent.text
                if (suspects[topic][0] == 'dea') and not DEA_FOUND:
                    if ent.label_ in Topics['dea']:
                        dea = ent.text
           #     if suspects[topic][0] == 'licence':  
            #        if ent.label_ in Topics['licence']:
             #           licensure = ent.text
            if suspects[topic][0] == 'outcome':
                outcome.append(sentence)
            if suspects[topic][0] == 'init_act':   
                aux = re.findall(open_exp, sentence)
                sentence = ' '.join(aux)
                init_act.append(sentence) if sentence != '' else None
            if suspects[topic][0] == 'act_basis': 
                aux = re.findall(open_exp, sentence)
                sentence = ' '.join(aux)
                act_basis.append(sentence) if sentence != '' else None
            if suspects[topic][0] == 'relevant':
                relevant = False
            if suspects[topic][0] == 'licence':
                licensure.append(re.findall(licensure_exp, sentence)[0])
                #aux = re.findall(licensure_exp, sentence)
                #sentence = ' '.join(aux)
                #licensure.append(sentence) if sentence != '' else None
               #licensure.append(,'', sentence)))
            if suspects[topic][0] == 'gender':
                if 'MALE' in sentence:
                    gender = 'MALE'
                elif 'FEMALE' in sentence:
                    gender = 'FEMALE'
                else:
                    gender = 'not found'

            if suspects[topic][0] == 'ssn':
                ssn = sentence[-4:]
            if suspects[topic][0] == 'subsequent_action':
                aux = re.findall(open_exp, sentence)
                sentence = ' '.join(aux)
                subsequent_action.append(sentence) if sentence != '' else None
            if suspects[topic][0] == 'type_of_adverse_action':
                aux = re.findall(open_exp, sentence)
                sentence = ' '.join(aux)
                type_of_adverse_action.append(sentence) if sentence != '' else None       
            if suspects[topic][0] == 'dea':
               dea.append(str(re.sub(alpha_exp,'', sentence)))
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
        dea = str(dea)[2:-1]
        subsequent_action = str(subsequent_action)[1:-1]
        type_of_adverse_action = str(type_of_adverse_action)[1:-1]
        try: 
            event_date = [min(event_date[1:])]
        except:
            event_date = [datetime(1900,1,1)]

        # print('.'*50)
        # print('Practitioner name:' + pract_name)
        # print('Process date:' + str(max(proc_date)))
        # print('Paymat date:' + str(max(paid_date)))
        # print('Event date:' + str(min(event_date)))
        # print('Total pract amount:' + str(total_amount))
        # print('Amount:' + str(amount))
        # print('Outcome:' + str(outcome))
        # print('Ent_name:' + str(ent_name))
        # print('Init_act:' + str(init_act))
        # print('Act_basis:' + str(act_basis))
        # print('NPI:' + NPI)
        # print('Licencure:' + licensure)
        # print('DCN:' + dcn_list[dcn])
        # print('Date of birth:' + str(max(date_of_birth)))
        # print('Gender: ' + gender)
        # print('SSN:' + ssn)
        # print('FEIN:' + FEIN)
        # print('Subsequent action: ' +subsequent_action)
        # print('Type of action: ' + type_of_adverse_action)
        # print('DEA: ' + DEA)


        if db:
            import lossrun_models
            print("Inserting to database LOSSRUN" )
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
                              npi = NPI, 
                              amount_pract = amount,
                              dcn = dcn_list[dcn]
                              )
        
            print("Inserting to database RESULTS (QA)")
            lossrun_models.insert_results( license = licensure,
                                npi = NPI,
                                dcn = dcn_list,
                                entity_name = ent_name,
                                practitioner_name = pract_name,
                                action_initial = init_act,
                                action_basis = act_basis,
                                event_outcome = outcome,
                                process_date = str(max(proc_date)),
                                payment_date = str(max(paid_date)),
                                event_day = str(min(event_date)),
                                amount_pract =  str(amount),
                                payment_total_amount = str(total_amount)
                                )
        aux = {
            "license": licensure,
            "npi": NPI,
            "dcn": dcn_list[dcn],
            "entity_name": str(ent_name),
            "practitioner_name": pract_name,
            "action_initial": str(init_act),
            "action_basis": str(act_basis),
            "event_outcome": str(outcome),
            "process_date": str(max(proc_date).strftime("%m/%d/%Y")),
            "payment_date": str(max(paid_date).strftime("%m/%d/%Y")),
            "event_day": str(min(event_date).strftime("%m/%d/%Y")),
            "amount_pract": str(amount),
            "payment_total_amount": str(total_amount),
            # "date_of_birth": str(max(date_of_birth)),
            # "gender": gender,
            # "ssn": ssn,
            # "fein": FEIN,
            # "subsequent_action": subsequent_action,
            # "type_of_adverse_action": type_of_adverse_action,
            # "dea": DEA
        }
        aux = lossrun.check_null(aux)

        data_results.append(aux)


    print('Ingection time: %f' %(time.time()- inject_time ))

    return data_results
    # END OF CODE
    # %%

