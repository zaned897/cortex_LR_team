"""Naive Bayes classifation for policy, claim, and other.

    In this code a NB clasificator is trained to determinate if 
    certain string that match specifics rules is a claim, a policy or non relevant. This way,
    the lossruns info extraction can associate the results with every claim, and each claim to 
    a policy:
    --policy 1
            |-claim1
                    |data(insured, dates,etc.)
            |-claim2
            ...
    -policy2
    ...

    date: feb 2021
    @author: Eduardo S. Raul V.

    ASYMM DEVELOPERS
"""
#%%         load dependencies
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd 

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


data = pd.read_csv('/home/zned897/Proyects/pdf_text_extractor/nowinsurance-loss-runs/test/test_data_NB_class_claim_policy.csv')

string = data['string']

c_alphas, c_nums = [], []
for element in string:
    c_alphas.append(consecutive_elements(element)[0])
    c_nums.append(consecutive_elements(element)[1])

#%%
data['c_aphas'] = c_alphas
data['c_nums'] = c_nums
#%%
features = data.drop(columns=['type', 'string'])
target = data['type']

train_data, test_data, train_target, test_target = train_test_split(features, target, test_size=0.5,random_state = 0)

gnb = GaussianNB()

gnb.fit(train_data, train_target)

pred_target = gnb.predict(test_data)

print('*'*50)
print('Acuracy of the model', accuracy_score(pred_target, test_target))
