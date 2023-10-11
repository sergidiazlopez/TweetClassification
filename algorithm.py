import numpy as np
import operator
from numpy import log
class algorithm:
    def __init__(self, verbose=False, laplace_smoothing=False, max_dictionary_size = -1):
        self.laplace_smoothing = laplace_smoothing
        self.verbose = verbose

        # Metrics
        self.accuracy = 0
        self.precision = [0, 0]
        self.recall = 0
        self.conf_matrix = [0, 0, 0, 0]

        # Apartat B
        self.max_dictionary_size = max_dictionary_size

    def generate_counts(self, x, y):
        """
        Genera un diccionari amb el nombre de vegades que apareix cada paraula clau en un tweet positiu o en negatiu
        """
        dictionary = {}

        for content, classification in zip(x, y):
            for word in content.split():
                if not dictionary.get(word):    # Si no te entrada al diccionari, la creem
                    dictionary[word] = [0, 0]
                dictionary[word][classification] += 1

        return dictionary

    def generate_probabilities(self, dictionary, y, class_elements):
        """
        Genera diccionari amb la probabilitat de que la paraula apareixi en cada una de les classes
        """
        if self.laplace_smoothing:
            s = 1
        else:
            s = 0

        for word in dictionary:
            word_count = dictionary[word]
            conditional_probability = [0, 0]
            for i in [0, 1]:
                conditional_probability[i] = ((word_count[i] + s) / (class_elements[i] + s * len(dictionary)))
            dictionary[word] = conditional_probability

        if self.max_dictionary_size > 0:
            dictionary_sorted = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
            # Create a new dictionary with the highest values
            dictionary = {}
            for i, (key, value) in enumerate(dictionary_sorted):
                if i == self.max_dictionary_size:
                    break
                dictionary[key] = value

        return dictionary

    def fit(self, x, y):
        _, class_elements = np.unique(y, return_counts=True)
        #print(class_elements)
        d = self.generate_counts(x, y)
        p = self.generate_probabilities(d, y, class_elements)
        if self.verbose:
            print(sorted(p.items(), key=operator.itemgetter(1), reverse=False))

        self.taula_counts = d
        self.taula_probabilitats = p
        self.class_elements = class_elements

    def classify(self, x):
        total = self.class_elements[0] + self.class_elements[1]

        # Multiplica per la probabilitat a priori
        probabilitat = [self.class_elements[0] / total, self.class_elements[1] / total]

        # Calcula productori de Likelihood
        for word in x.split():
            if self.taula_probabilitats.get(word):
                entrada = self.taula_probabilitats[word]
                probabilitat[0] *= entrada[0]
                probabilitat[1] *= entrada[1]

        # Fer predicció
        if probabilitat[0] > probabilitat[1]:
            return 0
        else:
            return 1

    def test_model(self, x_test, y_test):
        """
        Funció per a avaluar el model i obtenir estadístiques (Precision, recall, F1-score, accuracy)
        :param x_test: Conjunt de dades de prova
        :param y_test: Etiquetes de classificació de les dades
        """
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for x, y in zip(x_test, y_test):
            if self.verbose:
                print("Paraula: ", x)
                print("Class. real:", y)
            p = self.classify(x)
            if self.verbose:
                print("Predicció: ", p)

            if p == 1 and y == 1:
                true_positive += 1
            if p == 1 and y == 0:
                false_positive += 1
            if p == 0 and y == 1:
                false_negative += 1
            if p == 0 and y == 0:
                true_negative += 1

        # Calcular mètriques
        #print("------------")
        #print("Prediccions correctes: ", true_negative + true_positive)
        #print("Prediccions totals: ", len(x_test))

        self.conf_matrix = [true_negative, false_positive, false_negative, true_positive]

        self.accuracy = (true_positive+true_negative)/len(x_test)

        self.precision = [0,0]
        self.precision[0] = true_negative / (true_negative + false_negative)
        self.precision[1] = true_positive / (true_positive + false_positive)

        self.recall = true_positive / (true_positive + false_negative)



    def print_metrics(self):
        print("Accuracy: ", self.accuracy)
        print("Precision (class negative):", self.precision[0])
        print("Precision (class positive):", self.precision[1])
        print("Recall:", self.recall)

    def print_confusion_matrix(self):
        print("Confusion matrix:")
        print("---------")
        print("| " + str(self.conf_matrix[0]) + " | " + str(self.conf_matrix[1]) + " |")
        print("---------")
        print("| " + str(self.conf_matrix[2]) + " | " + str(self.conf_matrix[3]) + " |")
        print("---------")

    def score(self):
        """
        For compatibility with cross_val_score, testing
        :return: accuracy
        """
        return self.accuracy






