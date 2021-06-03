from rb.processings.encoders.bert import BertWrapper, Lang
import argparse
import pandas as pd
from tensorflow import keras
import numpy as np
import bert


train_file = "FactualVerificari/context_labeled_factual_big_train_2.csv"
test_file = "FactualVerificari/context_labeled_factual_big_test_2.csv"
test_file = "FactualVerificari/test.csv"

def read_dataframe(input_file):
    return pd.read_csv(input_file, encoding="utf-8")

def statements_to_list(statements, contexts):
    articles = []
    for i, statement in enumerate(statements):
        for j in range(5):
            articles.append((statement, contexts[j][i]))

    return articles

def run_bert_rb(
    statements_train,
    labels_train,
    contexts_train,
    statements_test,
    labels_test,
    contexts_test,
):
    bert_wrapper = BertWrapper(Lang.RO, max_seq_len=512, custom_model=True)
    inputs, bert_output = bert_wrapper.create_inputs_and_model()
    cls_output = bert_wrapper.get_output(bert_output, "cls")

    output = keras.layers.Dense(4, activation='softmax')(cls_output)
    model = keras.Model(inputs=inputs, outputs=output)

    optimizer = keras.optimizers.Adam(lr=1e-5)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    bert_wrapper.load_weights()

    articles_train = statements_to_list(statements_train, contexts_train)
    articles_test = statements_to_list(statements_test, contexts_test)

    feed_inputs_train = bert_wrapper.process_input(articles_train)
    feed_inputs_test = bert_wrapper.process_input(articles_test)

    model.fit(feed_inputs_train, np.asarray(labels_train), epochs=10)

    result = model.predict(feed_inputs_test, batch_size=32)
    print(result)
    print(f"Mean squared error train: {model.evaluate(feed_inputs_train, np.asarray(labels_train))}")
    print(f"Mean squared error test: {model.evaluate(feed_inputs_test, np.asarray(labels_test))}")

def main():
    dataframe_train = read_dataframe(train_file)
    dataframe_test = read_dataframe(test_file)

    statements_train = dataframe_train["text"].tolist()
    statements_test = dataframe_test["text"].tolist()
    labels_train = dataframe_train["label"].tolist()
    labels_test = dataframe_test["label"].tolist()

    contexts_train = []
    for i in range(5):
        contexts_train.append(dataframe_train[f"context_{i}"].tolist())

    contexts_test = []
    for i in range(5):
        contexts_test.append(dataframe_test[f"context_{i}"].tolist())

    run_bert_rb(
        statements_train,
        labels_train,
        contexts_train,
        statements_test,
        labels_test,
        contexts_test,
    )

if __name__ == "__main__":
    main()
