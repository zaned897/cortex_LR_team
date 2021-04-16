"""EXTRACT ALL THE CLAIMS AND POLICY NUMBERS AND GENERAL DATA
IN LOSSRUN REPORTS. 

the data extracted will be used in a Naive Bayes algorithm to determinate wich one
is a claim and wich one is a policy number

date: feb-2021
@authon: Eduardo S. Raul V.
"""

#%% LOAD DATA
from pdf2image import convert_from_path
from pytesseract import image_to_data as i2d
from os import listdir
from cv2 import blur 
from pytesseract import Output
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from string import punctuation
from numpy import array
from numpy import mean


def get_cpus(load = 1/2):
    """Return the half of threads in cpu 
    INPUT: (int) the number of 
    """

    try:
        THREADS = cpu_count()
    except:
        THREADS = 1
        print("Cannot identify the Threads number, Threads = 1")

    return int(THREADS * load)


def get_files(path, extention):
    """ Retrn the files names in a folder whit the same extention
    INPUT: (str) directory path
           (str) extention i.e., '.pdf'
    RETURN: (list)
    """

    return [path + file for file in listdir(path) if extention.upper() in file.upper()] 


def apply_ocr(image):
    """Apply the Optical Character recognition
        INPUT: images
    """

    image = array(image)
    image[image<=125]=0
    image[image>126] = 255
    image = blur(image, (2,2))
    
    return i2d(image, output_type=Output.DICT)

    

def isaclaim(claim=''):
    """Check if a string has the format of a claim number (AANN.-NNA)
    Iput: (string) suspect claim
    Output: (bool) True if has the claim format
    """

    # load elemets of reference, special chars and dates, excluded
    punkt = punctuation.replace('.','').replace('-','').replace('_','') # dot and hyphen allowed in claim numbers
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # properties of claims
    match_length = len(claim)>=7
    match_digits = any(char.isdigit() for char in claim)
    match_special_chars = not any(char in punkt for char in claim)
    match_no_dates = not any(month in claim.lower() for month in months)

    return match_length and match_digits and match_special_chars and match_no_dates


def consecutive_elements(string):
    """Count the consecutive alphanumeric and digits
    """

    c_char, c_nums = 0, 0

    for i in range(len(string) - 1):
        if string[i].isdigit() and string[i+1].isdigit():
            c_nums += 1
        elif string[i].isalpha() and string[i+1].isalpha():
            c_char += 1
    return c_char, c_nums



def count_chars(string):
    """Count elements in a string for databse
    """

    chars, nums, punkts = 0, 0, 0
    
    for char in string:
        if char.isalpha():
            chars +=1
        elif char.isdigit():
            nums += 1
        else:
            punkts += 1 
    return chars, nums, punkts 

def get_claims(file_list):
   """ Return a data set of claims and policy numbers
   INPUT: (list) list of files in a train/test folder
   RETURN: (list) list of list data set
   """ 
   # store the elements with claim/policy format

   file_list = [file_list] 
   claims_or_policy = []
   claims =[]
   policy = []
    # read every file 

   for file in file_list:
        # get a image list (one image for every page)
        images = convert_from_path(file)
       
        #store each text in a image in a dict
        dictionaries = []
        
        # apply multi-threads in ocr, each page is loaded in a single cpu
        with ThreadPoolExecutor(max_workers=get_cpus()) as executor:
           results = executor.map(apply_ocr, images)
           [dictionaries.append(result) for result in results]

        for __dict in dictionaries:
            for idx, word in enumerate(__dict['text']):
                if isaclaim(word):
                    x_1 = __dict['top'][idx]
                    y_1 = __dict['left'][idx]
                    delta = __dict['width'][idx]
                    chars, nums, punks = count_chars(word)
                    c_alphas, c_nums = consecutive_elements(word)
                    claims_or_policy.append([word, x_1, y_1, delta, chars, nums, punks, c_alphas, c_nums])

        length_average = mean([len(element[0]) for element in claims_or_policy])
        [claims.append(element) if len(element[0]) <= length_average else policy.append(element) for element in claims_or_policy]
   return claims, policy


def main():
    # workdirectory and load info 
    directory_path = '/home/zned897/Proyects/pdf_text_extractor/nowinsurance-loss-runs/docs/lossruns_feasibility/'
    extention = '.pdf'
    pdf_files = get_files(path = directory_path, extention = extention)

    # extract every suspect if match the 
    file = 2
    claims, policys = get_claims(pdf_files[file])
    print(pdf_files[file])
    print('****   CLAIMS **** ')
    print('.'*50)
    print(claims)
    print('****    POLICYS     *****')
    print('.'*50)
    print(policys)


if __name__ == '__main__':
    main()


