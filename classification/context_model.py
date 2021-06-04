from rb.processings.encoders.bert import BertWrapper, Lang
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow import keras

train_file = "FactualVerificari/context_labeled_factual_big_train_2.csv"
train_file = "FactualVerificari/test.csv"
test_file = "FactualVerificari/context_labeled_factual_big_test_2.csv"

def read_dataframe(input_file):
    return pd.read_csv(input_file, encoding="utf-8")

def statements_to_list(statements, contexts, labels):
    articles = []
    new_labels = []

    for i, statement in enumerate(statements):
        for j in range(5):
            articles.append((statement, contexts[j][i]))
        new_labels += [labels[i]] * 5

    return articles, np.array(new_labels, np.int32)

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
        contexts_train,
        statements_test,
        labels_test,
        contexts_test,
        cv=True
):
    batch_size = 16
    bw = BertWrapper(Lang.RO, max_seq_len=256)
    all_articles, labels_train = statements_to_list(statements_train, contexts_train, labels_train)
    all_inputs = bw.process_input(all_articles)
    sample_weights = []

    for target in labels_train:
        weight = 1.
        if target == 0:
            weight = 256. / 236
        elif target == 1:
            weight = 256. / 120
        elif target == 2:
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
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics="mae")
            es = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            history = model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=10,
                                validation_data=(dev_inputs, dev_outputs),
                                callbacks=[es])  # , sample_weight=train_weights)
            epoch, loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
            frozen_epochs.append(epoch + 1)

            model.layers[3].trainable = True
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", metrics="mae")
            es = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            history = model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=10,
                                validation_data=(dev_inputs, dev_outputs),
                                callbacks=[es])  # , sample_weight=train_weights)
            epoch, loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
            losses.append(loss)
            accuracies.append(history.history["val_accuracy"][epoch])
            finetune_epochs.append(epoch + 1)

        print(np.mean(frozen_epochs), np.mean(finetune_epochs), np.mean(losses), np.mean(accuracies))
    else:
        test_articles, labels_test = statements_to_list(statements_test, contexts_test, labels_test)
        test_inputs = bw.process_input(test_articles)
        accuracies = []
        for i in range(10):
            bw = BertWrapper(Lang.RO, max_seq_len=256)
            model = build_model(bw)
            model.layers[3].trainable = False
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
            model.fit(all_inputs, labels_train, batch_size=batch_size, epochs=7,
                      validation_data=(test_inputs, labels_test), sample_weight=sample_weights)
            # model.fit(all_inputs, labels_train, batch_size=batch_size, epochs=round(np.mean(frozen_epochs)), validation_data=(test_inputs, labels_test), sample_weight=sample_weights)
            model.layers[3].trainable = True
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", metrics=["mae"])
            history = model.fit(all_inputs, labels_train, batch_size=batch_size, epochs=1,
                                validation_data=(test_inputs, labels_test), sample_weight=sample_weights)
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
