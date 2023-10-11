# Practica 3, Coneixement Raonament i Incertesa
# Sergi Diaz Lopez
# NIU 1599349
import pandas as pd

from algorithm import *
from validation import *
import operator

import warnings
warnings.filterwarnings("ignore")

# Preprocessing
df = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv', sep=';', usecols=['tweetText','sentimentLabel'])
#print(df['tweetText'].isna().sum())
#print(df['sentimentLabel'].isna().sum())
df = df.dropna()

x = df['tweetText']
Y = df['sentimentLabel']

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.3, random_state=1)

# Generar diccionari de paraules clau
naiveBayes = algorithm(verbose=False)
naiveBayes.fit(x_train, y_train)

# Individual preditction
print(x_test[0], " --- ", y_test[0])
print("prediction: ", naiveBayes.classify(x_test[0]))

# Avaluar model
print('No smoothing:')
naiveBayes.test_model(x_test, y_test)
naiveBayes.print_metrics()
naiveBayes.print_confusion_matrix()

# Avaluar model (smoothing)
print('Laplace smoothing:')
naiveBayes = algorithm(verbose=False, laplace_smoothing=True)
naiveBayes.fit(x_train, y_train)
naiveBayes.test_model(x_test, y_test)
naiveBayes.print_metrics()
naiveBayes.print_confusion_matrix()

# Cross validation
print('Cross-validation, no smoothing')
cross_validation(x,Y)

print('Cross-validation, with laplace smoothing')
cross_validation(x,Y, laplace_smoothing=True)

# Apartat B

# Fixar el conjunt de train, però utilitzar diferents mides de diccionari
for size in [100, 500, 1000, 5000, 10000]:
    print(f'Limit diccionari, max size = {size}')
    naiveBayes = algorithm(verbose=False, laplace_smoothing=True, max_dictionary_size=size)
    naiveBayes.fit(x_train, y_train)
    naiveBayes.test_model(x_test, y_test)
    naiveBayes.print_metrics()
    naiveBayes.print_confusion_matrix()

# Ampliar el conjunt de train (podeu determinar vosaltres l’interval). A l’augmentar el
# nombre de tweets, el diccionari també canviarà de mida.
for k in [2, 3, 5, 7, 10, 50]:
    print(f'Cross-validation, with k={k}')
    cross_validation(x, Y, laplace_smoothing=True, n_splits=k)

# Utilitzar sempre la mateixa mida de diccionari, però modificant el conjunt de train,
# per veure com afecta això a l’entrenament.
for k in [2, 3, 5, 7, 10, 50]:
    for size in [100, 5000, 10000]:
        print(f'Cross-validation, with k={k}, size limit {size}')
        cross_validation(x, Y, laplace_smoothing=True, n_splits=k, max_dictionary_size=size)