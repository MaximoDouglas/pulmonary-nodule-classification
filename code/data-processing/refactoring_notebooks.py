import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file_name  = '../../data/features/solidNodules.csv'

df = pd.read_csv(file_name)
print(df.columns[3])

'''print("max before: " + str(np.max(df[df.columns[3]])))
df[df.columns[3]].plot.kde()
plt.legend(["DPVO"])'''

#scaler = MinMaxScaler(copy=False)
scaler = StandardScaler(copy=True)
X = scaler.fit_transform(df[df.columns[2:73]])
X = pd.DataFrame(X)

print("max after: " + str(np.max(X[X.columns[0]])))
X[X.columns[0]].plot.kde()
plt.legend(["DPVN"])

plt.ylabel("Densidade de Probabilidade")
plt.xlabel("Entropia")
plt.show()

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

'''c_expon_gen     = stats.expon(scale=100, loc=0)
gamma_expon_gen = stats.expon(scale=0.1, loc=0)
'''

'''gen = stats.norm(scale=0.05, loc=0.4)
x = np.linspace(0, 0.5, 100)

plt.plot(x, gen.pdf(x), 'b-')
plt.title('Dropout space Distribution')
plt.show()'''

'''indx = [2, 3, 4, 6, 9, 11, 12, 18, 19, 22, 26, 28, 29, 
      30, 31, 32, 33, 35, 38, 40, 41, 44, 46, 47, 48, 
      50, 53, 54, 56, 57, 59, 60, 62, 63, 65, 67, 71, 72]

for ind in indx:
  print(df.columns[ind], end=", ")

print(len(indx))'''

'''\item ATI: energia, entropia, kurtosis, intensidade média, intensidade mínima, raiz quadrada média, skewness
\item AF: desproporção esférica, esfericidade, relação superfícia volume 
\item AT a 0º: entropia, homogeneidade, correlação, matiz, proeminência, variância, momento da diferenca inverso 
\item AT a 135º: entropia, correlação, proeminência, variância 
\item AT a 45º: entropia, homogeneidade, correlação, matiz, variância 
\item AT a 90º: entropia, contraste, correlação, matiz, variância, momento da diferenca inverso
\item ANB: soma dos valores, soma dos quadrados, média aritmética, variância da população, medida de skewness, segundo momento central

'''