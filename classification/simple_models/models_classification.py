import pandas as pd
import contractions
import nltk
import pickle
import argparse
import time
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

FILENAME_TRAIN = "../politifact.csv"
FILENAME_TEST = "../factual.csv"
NEW_FILENAME_TRAIN = "politifact_processed.csv"
NEW_FILENAME_TEST = "factual_processed.csv"
MODELS_PATH = "Models/"

SVM = 'svm'
RF = 'rf'
NBAYES = 'nbayes'

def read_dataframe(filename):
	df = pd.read_csv(filename)
	return df

def save_dataframe(df, filename):
	df.to_csv(filename, encoding="utf-8", index=False)

def save_model(model, filename):
	pickle.dump(model, open(filename, 'wb'))

def vectorize_texts(texts, no_features):
	vectorizer = TfidfVectorizer(preprocessor=' '.join, max_features=no_features)
	features = vectorizer.fit_transform(texts)
	return (vectorizer, features)

def process_text(dataframe, column_name):
	texts = dataframe[column_name].tolist()
	marks = "„”" + punctuation

	for index in range(len(texts)):
		text = texts[index]
		text = contractions.fix(text)
		text = ' '.join([w.lower() for w in nltk.word_tokenize(text)])
		text = ''.join(l for l in text if l not in marks and not l.isdigit())
		text = text.strip()
		word_tokens = nltk.word_tokenize(text)
		texts[index] = word_tokens

	return texts

def print_time(start_time, end_time, process):
	diff = end_time - start_time
	if int(diff / 60) > 0:
		print ("Timp", process, diff / 60, " minute")
	else:
		print ("Timp", process, diff / 60, " secunde")

def get_scores(y_test, predictions):
	print ('Accuracy: ', accuracy_score(y_test, predictions))
	print ('Recall: ', recall_score(y_test, predictions, average='micro'))
	print ('Precision: ', precision_score(y_test, predictions, average='micro'))
	print ('F1 score: ', f1_score(y_test, predictions, average='micro'))

	return accuracy_score(y_test, predictions)

def get_model(x_train, y_train, x_test, y_test, model_type):
	start_time = time.time()

	if model_type == SVM:
		clf = SVC(gamma='scale', decision_function_shape='ovr')
	elif model_type == RF:
		clf = RandomForestClassifier(n_estimators=100, random_state=42)
	else:
		clf = MultinomialNB()

	clf.fit(x_train, y_train)
	save_model(clf, MODELS_PATH + model_type)
	end_time = time.time()
	print_time(start_time, end_time, "antrenare model " + model_type)

	start_time = time.time()
	predictions = clf.predict(x_test)
	accuracy = get_scores(y_test, predictions)
	end_time = time.time()
	print (classification_report(y_test, predictions, digits=3))
	print_time(start_time, end_time, "predict model " + model_type)

	return accuracy, predictions

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="rf", help="Choose model: svm, rf, nbayes")
	args = parser.parse_args()

	model_type = args.model

	if model_type != SVM and model_type != RF and model_type != NBAYES:
		print ("Choose model: svm, rf, nbayes")
		return

	df_train = read_dataframe(FILENAME_TRAIN)
	df_test = read_dataframe(FILENAME_TEST)

	processed_texts_train = process_text(df_train, 'text')
	df_train['processed_texts'] = processed_texts_train
	save_dataframe(df_train, MODELS_PATH + NEW_FILENAME_TRAIN)

	processed_texts_test = process_text(df_test, 'English')
	df_test['processed_texts'] = processed_texts_test
	save_dataframe(df_test, MODELS_PATH + NEW_FILENAME_TEST)

	print ("Texte procesate si salvate...")

	vectorizer, x_train = vectorize_texts(processed_texts_train, 1000)
	save_model(vectorizer, MODELS_PATH + "vectorizer_" + model_type)

	print ("Vectorizator antrenat...")

	x_test = vectorizer.transform(processed_texts_test)
	y_train = df_train['label']
	y_test = df_test['label']

	accuracy, predictions = get_model(x_train, y_train, x_test, y_test, model_type)

	df_test[model_type] = predictions
	save_dataframe(df_test, MODELS_PATH + NEW_FILENAME_TEST)

if __name__ == "__main__":
	main()