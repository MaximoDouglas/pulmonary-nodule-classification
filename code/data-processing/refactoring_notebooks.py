import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

'''file_name  = '../../data/features/solidNodules.csv'

df = pd.read_csv(file_name)

scaler = MinMaxScaler(copy=False)
X = scaler.fit_transform(df[df.columns[2:73]])
X = pd.DataFrame(X)

print(np.max(df[df.columns[3]]))
print(df.columns[3])

df[df.columns[3]].plot.kde()
plt.legend(["DPVO"])
#X[X.columns[1]].plot.kde()
#plt.legend(["DPVN"])
plt.ylabel("Densidade de Probabilidade")
plt.xlabel("Entropia")
plt.show()'''

'''malignancies = df[df.columns[-2]]

dist = malignancies.value_counts()

print(dist)

images_dir = '../../data/images/solid-nodules-with-attributes/'

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

sub = get_immediate_subdirectories(images_dir)

summary = {'benigno':[], 'maligno':[]}

for malignancy in sub:
  exams = get_immediate_subdirectories(images_dir + malignancy)
  
  for exam in exams:
    nodules = get_immediate_subdirectories(images_dir + malignancy + '/' + exam)
    
    for nodule in nodules:
      nodule_full_path = images_dir + malignancy + '/' + exam + '/' + nodule
      
      files = len([name for name in os.listdir(nodule_full_path)
            if os.path.isfile(os.path.join(nodule_full_path, name))])

      summary[malignancy].append(files)

df_ben = pd.DataFrame(summary['benigno'])
df_mal = pd.DataFrame(summary['maligno'])

fig, ax = plt.subplots()
ax.boxplot([summary['maligno'], summary['benigno']])
ax.set_xticklabels(['Malignos', 'Benignos'])
ax.yaxis.set_ticks(np.arange(0, 50, 2))
plt.show()

print("Df Ben min/max: " + str(df_ben[0].min()) + ' / ' + str(df_ben[0].max()))
print("Df Mal min/max: " + str(df_mal[0].min()) + ' / ' + str(df_mal[0].max()))

print("Df Ben: " + str(df_ben[0].median()) + ' (+/- ' + str(df_ben[0].std()) + ')')
print("Df Mal: " + str(df_mal[0].median()) + ' (+/- ' + str(df_mal[0].std()) + ')')'''

c_expon_gen     = stats.expon(scale=100, loc=0)
gamma_expon_gen = stats.expon(scale=0.1, loc=0)

x = np.linspace(0, 500, 500)

plt.plot(x, c_expon_gen.pdf(x), 'b-')
plt.title('Gamma space Distribution')
plt.show()