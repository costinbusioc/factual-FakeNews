from rb.processings.encoders.bert import BertWrapper, Lang
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import argparse
import pandas as pd
from tensorflow import keras
import numpy as np


train_files = ["FactualVerificari/labeled_factual_big_train_2.csv"]
test_files = ["FactualVerificari/labeled_factual_big_test_2.csv"]


def read_dataframe(input_file):
    dataframe = pd.read_csv(input_file, encoding="utf-8")
    dataframe = dataframe.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    return dataframe


def run_bert_rb(
        statements_train,
        labels_train,
        statements_test,
        labels_test,
        first_validation_pars_train=None,
        last_validation_pars_train=None,
        first_validation_pars_test=None,
        last_validation_pars_test=None,
):
    bert_wrapper = BertWrapper(Lang.RO, max_seq_len=256, custom_model=True)
    inputs, bert_output = bert_wrapper.create_inputs_and_model()
    cls_output = bert_wrapper.get_output(bert_output, "cls")

    output = keras.layers.Dense(1, activation='linear')(cls_output)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error')
    bert_wrapper.load_weights()

    articles_train = []
    if first_validation_pars_train and last_validation_pars_train:
        for index in range(len(statements_train)):
            article = statements_train[index]
            context = ""
            if isinstance(first_validation_pars_train[index], str):
                context += f"{first_validation_pars_train[index]} "
            if isinstance(last_validation_pars_train[index], str):
                context += last_validation_pars_train[index]

            articles_train.append((article, context))
    else:
        for index in range(len(statements_train)):
            articles_train.append(statements_train[index])

    articles_test = []
    if first_validation_pars_test and last_validation_pars_test:
        for index in range(len(statements_test)):
            article = statements_test[index]
            context = ""
            if isinstance(first_validation_pars_test[index], str):
                context += f"{first_validation_pars_test[index]} "
            if isinstance(last_validation_pars_test[index], str):
                context += last_validation_pars_test[index]
            articles_test.append((article, context))
    else:
        for index in range(len(statements_test)):
            articles_test.append(statements_test[index])

    feed_inputs_train = bert_wrapper.process_input(articles_train)
    feed_inputs_test = bert_wrapper.process_input(articles_test)

    model.fit(feed_inputs_train, np.asarray(labels_train))

    result = model.predict(feed_inputs_test, batch_size=32)
    print(result)
    print(f"Mean squared error test: {mean_squared_error(result, labels_test)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="train model")
    parser.add_argument("-p", "--predict", action="store_true", help="predict model")
    parser.add_argument("--feature-type", type=str, default="statement")

    args = parser.parse_args()

    feature_type = args.feature_type

    if feature_type != "with-validation" and feature_type != "statement":
        print(
            "Train with simple statements or statements and validation: statement vs with-validation"
        )
        exit(0)

    dataframe_train = read_dataframe(train_files[0])
    dataframe_test = read_dataframe(test_files[0])

    statements_train = dataframe_train["text"].tolist()
    first_validation_pars_train = dataframe_train["first_validation_par"].tolist()
    last_validation_pars_train = dataframe_train["last_validation_par"].tolist()
    label = dataframe_train["label"]
    labels_train = []
    for l in label:
        if l == 0:
            labels_train.append(0)
        elif l == 1:
            labels_train.append(0.33)
        elif l == 2:
            labels_train.append(0.66)
        else:
            labels_train.append(1)

    statements_test = dataframe_test["text"].tolist()
    first_validation_pars_test = dataframe_test["first_validation_par"].tolist()
    last_validation_pars_test = dataframe_test["last_validation_par"].tolist()
    label = dataframe_test["label"]
    labels_test = []
    for l in label:
        if l == 0:
            labels_test.append(0)
        elif l == 1:
            labels_test.append(0.33)
        elif l == 2:
            labels_test.append(0.66)
        else:
            labels_test.append(1)

    run_bert_rb(
        statements_train,
        labels_train,
        statements_test,
        labels_test,
        first_validation_pars_train,
        last_validation_pars_train,
        first_validation_pars_test,
        last_validation_pars_test,
    )

if __name__ == "__main__":
    main()
