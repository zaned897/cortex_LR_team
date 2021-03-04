# %% check the processor cores
"""
import multiprocessing
margin = 50
try:
    cores_availables = multiprocessing.cpu_count()
    print("Cores availables: %d" %cores_availables)
except: 
    print("No processor info available\n")
print("."*50)

# %% test regular expression
import re
# money expression
mon_example = '$ 7,395.00'
print(re.sub(r"[^0-9\.0-9]+",'', mon_example))


# %% IMAGE PROC COVER PAGE
import pytesseract as pt 
from pytesseract import Output
from pdf2image import convert_from_path
import concurrent.futures
import cv2
import numpy as np
import skimage.filters
image_path = 'NPDB/data/NPDBQA2.pdf'

images = convert_from_path(image_path, grayscale=True, dpi=300)

def extract_text(image): 
    # rescale image if necesary
    image_255 = cv2.blur(np.array(image),(2,2))
    return pt.image_to_data(image_255,output_type=Output.DICT)

dictionaries = []  

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor: 
    results = [executor.submit(extract_text, image) for image in images]
    for result in concurrent.futures.as_completed(results):
        dictionaries.append(result.result())
print("DONE...")
# %%
' '.join(dictionaries[0]['text'])

#%% 
import matplotlib.pyplot as plt
kernel =  np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1/64)
kernel_none = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
_, bin_image= cv2.threshold(np.array(images[1]), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
image_proto = cv2.blur(bin_image,(2,2))

plt.figure(figsize=(23,17))
plt.imshow(image_proto, cmap='gray')

# %% 
' '.join(pt.image_to_data(image_proto, output_type=Output.DICT)['text'])



# %% image enhancement by unsharped mask extended
image_gauss = cv2.GaussianBlur(np.array(images[1]),(3,3),2.0) *255

unsharp_rgb = cv2.addWeighted(np.array(images[0]), .5, image_gauss, -0.5, 0)


plt.figure(figsize=(23,17))
plt.imshow(image_gauss, cmap='gray')

' '.join(pt.image_to_data(image_gauss, output_type=Output.DICT)['text'])

# check image as b/w
# %%
#image_base= np.array(images[0])
#_, image_bin =  cv2.threshold(np.array(images[1]), 126, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#image_bin = cv2.(image_bin - 1,(3,3))
#plt.figure(figsize=(23,17))
#plt.imshow(image_bin, cmap='gray')
#' '.join(pt.image_to_data(image_bin, output_type=Output.DICT)['text'])
"""
# %% 
import getopt, sys
args_list = sys.argv[1:]
args_flags = 'hf:m:d:'
args_long = ['help', 'file-path=','model-path=', 'data-base-insert=']
print("cheking the sys argumetns...")
argv = sys.argv[1:]

try:
   argument, values = getopt.getopt(args_list, args_flags, args_long)
   print('Reading argumets..')
except getopt.error as err: 
    print(str(err))
    sys.exit(2)
print('.'*10)
print(argument)
print('.'*10)
for current_argument, current_value in argument:
    if current_argument in ['-h','--help']:
        print(current_argument)
        print('Display help...')
    elif current_argument in ['-f', '--file-path']:
        print('Processing file: '+ str(current_value))
    elif current_argument in ['-m', '--model-path']:
        print('Model directory: ' + str(current_value))
    elif current_argument in ['-d', '--data-base-insert']:
        db = [False, True][current_value.lower()[0] == 't' or current_value.lower()[0]=='y']
        print(db)
        if db:
            print('inserting database available...')

#%%
test ='Frue'
test = [False, True][test.lower()[0] == 't' or test.lower()[0]=='y']
test



# %% 
from configobj import ConfigObj

topics = ConfigObj('NPDB/config/config_npdb_topics.ino')

for key, item in enumerate(topics):
    print(item)

#%%
X = ['1','2']
X =+ ['0']