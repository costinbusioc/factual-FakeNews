from rb.processings.encoders.bert import BertWrapper, Lang
import argparse
import pandas as pd
from tensorflow import keras
import numpy as np
import bert


train_files = ["FactualVerificari/labeled_factual_big_train_2.csv"]
test_files = ["FactualVerificari/labeled_factual_big_test_2.csv"]


def read_dataframe(input_file):
    dataframe = pd.read_csv(input_file, encoding="utf-8")
    dataframe = dataframe.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    return dataframe

def convert_labels_for_regression(labels):
    reg_labels = []
    for l in labels:
        if l == 0:
            reg_labels.append(0)
        elif l == 1:
            reg_labels.append(0.33)
        elif l == 2:
            reg_labels.append(0.66)
        else:
            reg_labels.append(1)

    return reg_labels

def statements_to_list(statements, firs_validation, last_validation):
    articles = []
    if firs_validation and last_validation:
        for index in range(len(statements)):
            article = statements[index]
            context = ""
            if isinstance(firs_validation[index], str):
                context += f"{firs_validation[index]} "
            if isinstance(last_validation[index], str):
                context += last_validation[index]
            articles.append((article, context))
    else:
        for index in range(len(statements)):
            articles.append(statements[index])

    return articles

def run_bert_rb(
        statements_train,
        labels_train,
        statements_test,
        labels_test,
        first_validation_pars_train=None,
        last_validation_pars_train=None,
        first_validation_pars_test=None,
        last_validation_pars_test=None,
        freeze=False,
):
    bert_wrapper = BertWrapper(Lang.RO, max_seq_len=256, custom_model=True)
    inputs, bert_output = bert_wrapper.create_inputs_and_model()
    cls_output = bert_wrapper.get_output(bert_output, "cls")

    output = keras.layers.Dense(1, activation='linear')(cls_output)
    model = keras.Model(inputs=inputs, outputs=output)

    if freeze:
        for layer in model.layers:
            if isinstance(layer, bert.BertModelLayer):
                layer.trainable = False

    optimizer = keras.optimizers.Adam(lr=1e-5)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
    bert_wrapper.load_weights()

    articles_train = statements_to_list(statements_train, first_validation_pars_train, last_validation_pars_train)
    articles_test = statements_to_list(statements_test, first_validation_pars_test, last_validation_pars_test)

    feed_inputs_train = bert_wrapper.process_input(articles_train)
    feed_inputs_test = bert_wrapper.process_input(articles_test)

    model.fit(feed_inputs_train, np.asarray(labels_train), epochs=10)

    # result = model.predict(feed_inputs_test, batch_size=32)
    print(f"Mean squared error train: {model.evaluate(feed_inputs_train, np.asarray(labels_train))}")
    print(f"Mean squared error test: {model.evaluate(feed_inputs_test, np.asarray(labels_test))}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", default=False, dest="freeze")
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
    labels_train = convert_labels_for_regression(dataframe_train["label"])

    statements_test = dataframe_test["text"].tolist()
    labels_test = convert_labels_for_regression(dataframe_test["label"])

    first_validation_pars_train, last_validation_pars_train = None, None
    first_validation_pars_test, last_validation_pars_test = None, None
    if feature_type == "with-validation":
        first_validation_pars_train = dataframe_train["first_validation_par"].tolist()
        last_validation_pars_train = dataframe_train["last_validation_par"].tolist()

        first_validation_pars_test = dataframe_test["first_validation_par"].tolist()
        last_validation_pars_test = dataframe_test["last_validation_par"].tolist()

    run_bert_rb(
        statements_train,
        labels_train,
        statements_test,
        labels_test,
        first_validation_pars_train,
        last_validation_pars_train,
        first_validation_pars_test,
        last_validation_pars_test,
        args.freeze,
    )

if __name__ == "__main__":
    main()
