"""Unit test to evaluate the nlp pipeline and entiry recognition
    
    A large model is used as reference
"""
# %% LOAD DEPENDNCIES
from pdf2image import convert_from_path
import pytesseract as pt
from pytesseract import Output
import spacy


# %%        LOAD MODELS
nlp_base = spacy.load('/home/zned897/Proyects/pdf_text_extractor/external_models/NPDB_models/NPDB_ner_model')
nlp_test = spacy.load('/home/zned897/Proyects/pdf_text_extractor/nowinsurance-loss-runs/models/base_model')



# %% TEST COMMON SENTENCES IN LOSS REPORTS

ORG = 'd4 inc'      # first organization in lossrun reports

doc1 = nlp_base(ORG)# apply NER with the large model
doc2 = nlp_base(ORG)      # apply light model

for ent in doc1.ents:
    print(ent.label_, ent.text)

for ent in doc2.ents:
    print(ent.label_, ent.text)
