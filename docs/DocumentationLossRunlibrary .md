# Documentation Loss Run library
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Methods</summary>
  <ol>
    <li><a href="#update_files_in_path">update_files_in_path()</a></li>
    <li><a href="#transform_to_images_an_entire_folder">transform_to_images_an_entire_folder()</a></li>
    <li><a href="#transform_to_text_an_entire_folder">transform_to_text_an_entire_folder()</a></li>
    <li><a href="#spatial_filter">spatial_filter()</a></li>
    <li><a href="#pre_proc">pre_proc()</a></li>
    <li><a href="#read_dict">read_dict()</a></li>
    <li><a href="#extract_statistic_featrues">extract_statistic_featrues</a></li>
    <li><a href="#map_words">map_words()</a></li>
    <li><a href="#search_rules">search_rules()</a></li>
    <li><a href="#is_report">is_report()</a></li>
    <li><a href="#non_info_filter">non_info_filter()</a></li>
    <li><a href="#print_help">print_help()</a></li>
  </ol>
</details>

## update_files_in_path

```sh
   update_files_in_path(root=String,log_file=String)
```
**Check if files are added or erased from a folder**. This method get the path (string) of root folder and check if files are missing or added modified files are stored in the log_file
```sh
   return List
```

## transform_to_images_an_entire_folder
```sh
   transform_to_images_an_entire_folder(pdfs_folder = String, images_folder = String, format = String, log_file = String) 
```

**Transform all .pdf files in a folder to imges**. pdfs_folder get folder path containing all pdf reports,images_folder  get folder path where results were stored, format get target format for images, log_file get the name file

```sh
	return Boolean
```

## transform_to_text_an_entire_folder
```sh
   transform_to_text_an_entire_folder(images_folder = String , text_folder = String , save_string = Boolean, log_file=String)
```
**Transform all images files in a folder to text (dictionary format supported)**. images_folder get folder path containing all pdf reports, text_folder get folder path where results were stored, save_string define the format in wich the OCR will save the image analisys, log_file get the name file
```sh
   return Boolean
```
## spatial_filter
```sh
   spatial_filter(txt_dict=Dict, topics=ConfigObj, report_type=String)
```
**Catch the text in dictionary according to spatial(x, y) relation**. txt_dict get dictionary with raw txt info in pdf report,topics (list) get topics list of the entities of interes, report_type get type of report
```sh
   return List
```
## pre_proc
```sh
   pre_proc(pdf_file=String, data_path=String, topic_file=String, image_format = String, text_format = String)
```
**Creates an image with the founded topics enhanced**. pdf_file get PDF file name, data_path get data path same level than txt and images, topic_file get data path of topics, image_format get format image, text_format get format text
```sh
   return Dict,List,Image,Image
```
## read_dict
```sh
   read_dict(txt_file_path=String)
```
**Read txt file as dictionary**. txt_file_path get file location
```sh
   return String
```
## extract_statistic_featrues
```sh
   extract_statistic_featrues(list_of_paths_of_txt_files=List)
```
**Extract features from dictionary**. list_of_paths_of_txt_files get list of multiple txt files as input
```sh
   return np.array
```
## map_words
```sh
   map_words(txt_dict=Dict)
```
**Match items of words in text dictionary (map a string items in dictionary format)**. txt_dict get dictionary with raw txt info in pdf report
```sh
   return String,List
```
## search_rules
```sh
   search_rules(dictionary=Dict, rules=ConfigObj)
```
**Search the topics in the dictionary, several formats supported**.dictionary get dictionary with raw txt info in pdf report, rules get configuration file
```sh
   return List
```
## is_report
```sh
   is_report(image=np.array, txt=Dict)
```
**Determinate if the interest file is a lossrun, email or NPDB file based in the text content**. image get image in numpy format,txt get dictionary extracted from OCR stage
```sh
   return String
```
## non_info_filter
```sh
   non_info_filter(pdf_path=String, resize_factor = (Int,Int), score = Float)
```
**Evaluate if a page has relevant information based in image similarity**. pdf_path get string of the pdf file,resize_factor get smaller resize make faster the evaluation but less accurate,score get similitud scroe porcent
```sh
   return List[bool]
```
<!---## print_help
```sh
   print_help()
```
Extract relevant info from Loss Run reports applying OCR and NLP

```sh
   return None
```
>