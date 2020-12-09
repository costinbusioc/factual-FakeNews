import pandas as pd


INPUT_FILES_LIST = ["factual_covid_predictions_bert.csv", "factual_covid_predictions_xlmroberta.csv",
					"factual_predictions_bert.csv", "factual_predictions_xlmroberta.csv"]


def read_df(input_file):
    df = pd.read_csv(input_file)
    df = df.drop(['Unnamed: 0'], axis=1)
    return df


def main():
	for file in INPUT_FILES_LIST:
		df = read_df(file)
		labels = df['labels'].tolist()
		predictions = df['predictions'].tolist()
		total_test_size = len(labels)
		
		print ("\nDataset statistics: ", file)
		print ("Dataset size: ", total_test_size)
		for i in range(4):
			predictions_label_0 = 0
			predictions_label_1 = 0
			predictions_label_2 = 0
			predictions_label_3 = 0
			label_list_size = len([label for label in labels if label == i])
			print ("Label ", i, ": ", label_list_size, "/", total_test_size, " = ", label_list_size / total_test_size * 100, "%")
			print ("Label size: ", label_list_size)
			for index in range(len(labels)):
				if labels[index] == i:
					if predictions[index] == 0:
						predictions_label_0 += 1
					elif predictions[index] == 1:
						predictions_label_1 += 1
					elif predictions[index] == 2:
						predictions_label_2 += 1
					elif predictions[index] == 3:
						predictions_label_3 += 1
			if i == 0:
				print ("Correct predictions:    ", predictions_label_0, "/", label_list_size, " = ", predictions_label_0 / label_list_size * 100, "%")
				print ("Wrong predictions as 1: ", predictions_label_1, "/", label_list_size, " = ", predictions_label_1 / label_list_size * 100, "%")
				print ("Wrong predictions as 2: ", predictions_label_2, "/", label_list_size, " = ", predictions_label_2 / label_list_size * 100, "%")
				print ("Wrong predictions as 3: ", predictions_label_3, "/", label_list_size, " = ", predictions_label_3 / label_list_size * 100, "%")
			elif i == 1:
				print ("Correct predictions:    ", predictions_label_1, "/", label_list_size, " = ", predictions_label_1 / label_list_size * 100, "%")
				print ("Wrong predictions as 0: ", predictions_label_0, "/", label_list_size, " = ", predictions_label_0 / label_list_size * 100, "%")
				print ("Wrong predictions as 2: ", predictions_label_2, "/", label_list_size, " = ", predictions_label_2 / label_list_size * 100, "%")
				print ("Wrong predictions as 3: ", predictions_label_3, "/", label_list_size, " = ", predictions_label_3 / label_list_size * 100, "%")
			elif i == 2:
				print ("Correct predictions:    ", predictions_label_2, "/", label_list_size, " = ", predictions_label_2 / label_list_size * 100, "%")
				print ("Wrong predictions as 0: ", predictions_label_0, "/", label_list_size, " = ", predictions_label_0 / label_list_size * 100, "%")
				print ("Wrong predictions as 1: ", predictions_label_1, "/", label_list_size, " = ", predictions_label_1 / label_list_size * 100, "%")
				print ("Wrong predictions as 3: ", predictions_label_3, "/", label_list_size, " = ", predictions_label_3 / label_list_size * 100, "%")
			elif i == 3:
				print ("Correct predictions:    ", predictions_label_3, "/", label_list_size, " = ", predictions_label_3 / label_list_size * 100, "%")
				print ("Wrong predictions as 0: ", predictions_label_0, "/", label_list_size, " = ", predictions_label_0 / label_list_size * 100, "%")
				print ("Wrong predictions as 1: ", predictions_label_1, "/", label_list_size, " = ", predictions_label_1 / label_list_size * 100, "%")
				print ("Wrong predictions as 2: ", predictions_label_2, "/", label_list_size, " = ", predictions_label_2 / label_list_size * 100, "%")
			print ("=========================")


if __name__ == "__main__":
	main()