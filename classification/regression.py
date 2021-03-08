from transformers import BertTokenizer, BertModel
from rb.processings.encoders.bert import BertWrapper, Lang
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import pickle
import argparse
import pandas as pd
from tensorflow import keras


train_files = ["FactualVerificari/labeled_factual_big_train_2.csv"]
test_files = ["FactualVerificari/labeled_factual_big_test_2.csv"]


def read_dataframe(input_file):
    dataframe = pd.read_csv(input_file, encoding="utf-8")
    dataframe = dataframe.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    return dataframe


def get_features(statements, first_validation_pars=None, last_validation_pars=None):
    bert_wrapper = BertWrapper(Lang.RO, max_seq_len=128, custom_model=True)
    inputs, bert_output = bert_wrapper.create_inputs_and_model()
    cls_output = bert_wrapper.get_output(bert_output, "cls")

    model = keras.Model(inputs=inputs, outputs=[cls_output])
    model.compile()
    bert_wrapper.load_weights()

    articles = []
    if first_validation_pars and last_validation_pars:
        for index in range(len(statements)):
            article = statements[index]
            context = ""
            if isinstance(first_validation_pars[index], str):
                context += f"{first_validation_pars[index]} "
            if isinstance(last_validation_pars[index], str):
                context += last_validation_pars[index]

            articles.append((article, context))
    else:
        for index in range(len(statements)):
            articles.append(statements[index])

    feed_inputs = bert_wrapper.process_input(articles)
    return model.predict(feed_inputs, batch_size=32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="train model")
    parser.add_argument("-p", "--predict", action="store_true", help="predict model")
    parser.add_argument("--feature-type", type=str, default="statement")

    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        exit(0)

    feature_type = args.feature_type

    if feature_type != "with-validation" and feature_type != "statement":
        print(
            "Train with simple statements or statements and validation: statement vs with-validation"
        )
        exit(0)

    if args.train:
        dataframe = read_dataframe(train_files[0])
    if args.predict:
        dataframe = read_dataframe(test_files[0])

    statements = dataframe["text"].tolist()
    first_validation_pars = dataframe["first_validation_par"].tolist()
    last_validation_pars = dataframe["last_validation_par"].tolist()
    label = dataframe["label"]
    labels = []
    for l in label:
        if l == 0:
            labels.append(0)
        elif l == 1:
            labels.append(0.33)
        elif l == 2:
            labels.append(0.66)
        else:
            labels.append(1)

    if feature_type == "statement":
        features = get_features(statements)
        model_name = "RegressionModels/regression_statements"
    else:
        features = get_features(statements, first_validation_pars, last_validation_pars)
        model_name = "RegressionModels/regression_statements_with_validation"

    print(features[0])
    if args.train:
        print("Training set: ", len(labels))
        print("0 (Adevărat): ", len([label for label in labels if label == 0]))
        print(
            "0.33 (Parțial Adevărat): ",
            len([label for label in labels if label == 0.33]),
        )
        print(
            "0.66 (Parțial Fals): ", len([label for label in labels if label == 0.66])
        )
        print("1 (Fals): ", len([label for label in labels if label == 1]))
        reg = LinearRegression()
        reg.fit(features, labels)
        with open(model_name, "wb") as file:
            pickle.dump(reg, file)
        print("Model salvat: ", model_name)
        predictions = reg.predict(features)
        score = mean_squared_error(labels, predictions)
        print("Mean squarred error train: ", score)

    if args.predict:
        with open(model_name, "rb") as file:
            reg = pickle.load(file)
        predictions = reg.predict(features)
        for index in range(len(predictions)):
            print("Predictie: ", predictions[index])
            print("Real: ", labels[index])
        score = mean_squared_error(labels, predictions)
        print("Mean squarred error test: ", score)


if __name__ == "__main__":
    main()
