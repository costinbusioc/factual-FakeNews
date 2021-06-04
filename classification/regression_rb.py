import argparse
import csv
from collections import Counter

import numpy as np
import pandas as pd
from rb.processings.encoders.bert import BertWrapper, Lang
from sklearn.model_selection import KFold
from tensorflow import keras

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

    return np.array(reg_labels)

def statements_to_list(statements, first_validation, last_validation):
    articles = []
    if first_validation or last_validation:
        for index in range(len(statements)):
            article = statements[index]
            context = ""
            if first_validation and isinstance(first_validation[index], str):
                context += f"{first_validation[index]} "
            if last_validation and isinstance(last_validation[index], str):
                context += last_validation[index]
            articles.append((article, context))
    else:
        for index in range(len(statements)):
            articles.append(statements[index])

    return articles

def build_model(bw: BertWrapper) -> keras.Model:
    inputs, bert_output = bw.create_inputs_and_model()
    cls_output = bw.get_output(bert_output, "cls")
    cls_output = keras.layers.Dropout(0.2)(cls_output)
    hidden = keras.layers.Dense(16, activation='tanh')(cls_output)
    output = keras.layers.Dense(4, activation='softmax')(hidden)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def run_bert_rb(
        statements_train,
        labels_train,
        statements_test,
        labels_test,
        first_validation_pars_train=None,
        last_validation_pars_train=None,
        first_validation_pars_test=None,
        last_validation_pars_test=None,
        cv=True
):
    batch_size = 16
    bw = BertWrapper(Lang.RO, max_seq_len=256)
    all_articles = statements_to_list(statements_train, first_validation_pars_train, last_validation_pars_train)
    all_inputs = bw.process_input(all_articles)
    sample_weights = []
    for target in labels_train:
        weight = 1.
        if target == 0:
            weight = 256. / 236
        elif target == 1:
        # elif target == 0.33:
            weight = 256. / 120
        elif target == 2:
        # elif target == 0.66:
            weight = 256. / 62
        sample_weights.append(weight)
    sample_weights = np.array(sample_weights)
    if cv:
        model = build_model(bw)
        initial_weights = model.get_weights()
        kf = KFold(5, shuffle=True)
        losses = []
        accuracies = []
        frozen_epochs = []
        finetune_epochs = []
        for train_index, dev_index in kf.split(all_inputs[0]):
            train_inputs = [input[train_index] for input in all_inputs]
            dev_inputs = [input[dev_index] for input in all_inputs]
            train_outputs = labels_train[train_index]
            train_weights = sample_weights[train_index]
            dev_outputs = labels_train[dev_index]
            model.set_weights(initial_weights)
            model.layers[3].trainable = False
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics="mae")
            es = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            history = model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=10, validation_data=(dev_inputs, dev_outputs), callbacks=[es]) #, sample_weight=train_weights)
            epoch, loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
            frozen_epochs.append(epoch + 1)
            
            model.layers[3].trainable = True
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", metrics="mae")
            es = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            history = model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=10, validation_data=(dev_inputs, dev_outputs), callbacks=[es]) #, sample_weight=train_weights)
            epoch, loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
            losses.append(loss)
            accuracies.append(history.history["val_accuracy"][epoch])
            finetune_epochs.append(epoch + 1)

        print(np.mean(frozen_epochs), np.mean(finetune_epochs), np.mean(losses), np.mean(accuracies))
    else:
        test_articles = statements_to_list(statements_test, first_validation_pars_test, last_validation_pars_test)
        test_inputs = bw.process_input(test_articles)
        accuracies = []
        for i in range(10):
            bw = BertWrapper(Lang.RO, max_seq_len=256)
            model = build_model(bw)
            model.layers[3].trainable = False
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
            model.fit(all_inputs, labels_train, batch_size=batch_size, epochs=7, validation_data=(test_inputs, labels_test), sample_weight=sample_weights)
            # model.fit(all_inputs, labels_train, batch_size=batch_size, epochs=round(np.mean(frozen_epochs)), validation_data=(test_inputs, labels_test), sample_weight=sample_weights)
            model.layers[3].trainable = True
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", metrics=["mae"])
            history = model.fit(all_inputs, labels_train, batch_size=batch_size, epochs=1, validation_data=(test_inputs, labels_test), sample_weight=sample_weights)
            accuracies.append(history.history["val_accuracy"][-1])
        print(f"Min acc: {np.min(accuracies)}")
        print(f"Max acc: {np.max(accuracies)}")
        print(f"Mean acc: {np.mean(accuracies)}")
    
    
        # predictions = model.predict(test_inputs)
        # confusion = np.zeros((4,4), dtype=np.int32)
        # for pred, target in zip(predictions, labels_test):
        #     pred = int(np.argmax(pred))
        #     # pred = round(pred[0] * 3)
        #     # target = round(target * 3)
        #     confusion[pred, target] += 1
        # print(confusion)
        # with open("predictions.csv", "wt", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     for pred, target, text in zip(predictions, labels_test, statements_test):
        #         writer.writerow([int(np.argmax(pred)), target, text])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-type", type=str, default="statement")
    parser.add_argument("--cv", action="store_true")

    args = parser.parse_args()

    feature_type = args.feature_type
    cv = args.cv

    if feature_type != "with-validation" and feature_type != "statement":
        print(
            "Train with simple statements or statements and validation: statement vs with-validation"
        )
        exit(0)

    dataframe_train = read_dataframe(train_files[0])
    dataframe_test = read_dataframe(test_files[0])
    statements_train = dataframe_train["text"].tolist()
    labels_train = np.array(dataframe_train["label"], np.int32)
    # labels_train = convert_labels_for_regression(dataframe_train["label"])
    statements_test = dataframe_test["text"].tolist()
    labels_test = np.array(dataframe_test["label"], np.int32)
    # labels_test = convert_labels_for_regression(dataframe_test["label"])

    first_validation_pars_train, last_validation_pars_train = None, None
    first_validation_pars_test, last_validation_pars_test = None, None
    if feature_type == "with-validation":
        first_validation_pars_train = dataframe_train["first_validation_par"].tolist()
        # last_validation_pars_train = dataframe_train["last_validation_par"].tolist()

        first_validation_pars_test = dataframe_test["first_validation_par"].tolist()
        # last_validation_pars_test = dataframe_test["last_validation_par"].tolist()

    run_bert_rb(
        statements_train,
        labels_train,
        statements_test,
        labels_test,
        first_validation_pars_train,
        last_validation_pars_train,
        first_validation_pars_test,
        last_validation_pars_test,
        cv=cv
    )

if __name__ == "__main__":
    main()
