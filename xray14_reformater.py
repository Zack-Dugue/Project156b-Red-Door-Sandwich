import pandas as pd
import ast

findings_list = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other','Fracture','Support Devices']

#xray14
def xray14path(image):
    return "data/xray-14/images/"+image

def xray14sex(gender):
    assert(gender=='M' or gender=='F')
    if gender=='M':
        return 'Male'
    else:
        return 'Female'
    
def xray14findings(labels):
    label_arr = labels.split('|')
    new_arr = []
    for item in findings_list:
        if item in label_arr:
            new_arr.append(1)
        else:
            new_arr.append(-1)
    
    return new_arr

def xray14dropfunc(labels):
    if 1 in labels:
        return 1
    else:
        return 0

xray14 = pd.read_csv("xray14.csv")

xray14['Image Index'] = xray14['Image Index'].apply(xray14path)
xray14['Patient Gender'] = xray14['Patient Gender'].apply(xray14sex)

print(xray14)

new_xray14 = pd.DataFrame(columns=['filepath', 'sex', 'age', 'frontal/lateral', 'AP/PA', 'findings', 'drop_id'])
new_xray14['filepath'] = xray14['Image Index']
new_xray14['sex'] = xray14['Patient Gender']
new_xray14['age'] = xray14['Patient Age']
new_xray14['frontal/lateral'] = 'Frontal'
new_xray14['AP/PA'] = xray14['View Position']
new_xray14['findings'] = xray14['Finding Labels'].apply(xray14findings)

new_xray14['drop_id'] = new_xray14['findings'].apply(xray14dropfunc)

print(new_xray14)

new_xray14.drop(new_xray14[new_xray14['drop_id']==0].index, inplace=True)
new_xray14.drop('drop_id', axis=1, inplace=True)

print(new_xray14)

new_xray14.to_csv('xray14labels.csv', index=False)
