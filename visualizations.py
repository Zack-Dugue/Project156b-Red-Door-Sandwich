from matplotlib import pyplot as plt
import csv
# Unnamed: 0,Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
ID = 0
Path = 1
Sex = 2
Age = 3
Frontal_Lateral = 4
AP_PA = 5
No_Finding = 6

yes_finding = []
maybe_finding = []
no_finding = []

with open('labels.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for (i,row) in enumerate(reader):
        age = float(row[Age])
        if(row[No_Finding] == ''):
            continue
        if float(row[No_Finding]) == -1.0:
            yes_finding.append(age)
        elif float(row[No_Finding]) == 0.0:
            maybe_finding.append(age)
        elif float(row[No_Finding]) == 1.0:
            no_finding.append(age)
        print(f"\rrow : {i}", end="")

plt.hist(yes_finding,bins=[3*x for x in range(35)])
plt.title("Age Histogram (subset - something was found)")
plt.show()

plt.hist(no_finding,bins=[3*x for x in range(35)])
plt.title("Age Histogram (subset - something not found)")
plt.show()

plt.hist(maybe_finding,bins=[3*x for x in range(35)])
plt.title("Age Histogram (subset - something was maybe found)")
plt.show()
print("end")