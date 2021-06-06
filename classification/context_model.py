from rb.processings.encoders.bert import BertWrapper, Lang
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow import keras
from statistics import mode

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

def compute_values_based_on_max(results):
    max_vals, max_pos = [], []
    for result in results:
        max_val = np.max(result)
        pos = int(np.argmax(result))

        max_vals.append(max_val)
        max_pos.append(pos)

    return max_pos[max_vals.index(max(max_vals))]

def compute_values_based_on_majority(results):
    max_pos = []
    for result in results:
        max_pos.append(int(np.argmax(result)))

    return mode(max_pos)

def run_bert_rb(
        statements_train,
        labels_train,
        contexts_train,
        statements_test,
        labels_test,
        contexts_test,
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

    test_articles, labels_test = statements_to_list(statements_test, contexts_test, labels_test)
    test_inputs = bw.process_input(test_articles)

    accuracies = []
    max_accuracies = []
    most_common_accuracies = []

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

        predictions = model.predict(test_inputs)

        correct_max = 0
        correct_common = 0

        for pos in range(0, len(predictions), 5):
            statement_pred = predictions[pos:(pos+5)]

            max_result = compute_values_based_on_max(statement_pred)
            most_common = compute_values_based_on_majority(statement_pred)

            if max_result == labels_test[pos]:
                correct_max +=1

            if most_common == labels_test[pos]:
                correct_common += 1

            print(max_result)
            print(most_common)
            print(statement_pred)
            print("=======")

        max_accuracies.append((correct_max/(len(labels_test)/5)))
        most_common_accuracies.append((correct_common/(len(labels_test)/5)))

    print(f"Min acc: {np.min(accuracies)}")
    print(f"Max acc: {np.max(accuracies)}")
    print(f"Mean acc: {np.mean(accuracies)}")
    print(f"Min acc max: {np.min(max_accuracies)}")
    print(f"Max acc max: {np.max(max_accuracies)}")
    print(f"Mean acc max: {np.mean(max_accuracies)}")
    print(f"Min acc most common: {np.min(most_common_accuracies)}")
    print(f"Max acc most common: {np.max(most_common_accuracies)}")
    print(f"Mean acc most common: {np.mean(most_common_accuracies)}")

    predictions = model.predict(test_inputs)

    max_results = []
    common_results = []
    true_labels = []

    for pos in range(0, len(predictions), 5):
        statement_pred = predictions[pos:(pos + 5)]

        max_results.append(compute_values_based_on_max(statement_pred))
        common_results.append(compute_values_based_on_majority(statement_pred))
        true_labels.append(labels_test[pos])

    compute_confusion(max_results, true_labels)
    compute_confusion(common_results, true_labels)

def compute_confusion(predictions, true_labels):
    confusion = np.zeros((4,4), dtype=np.int32)
    for pred, target in zip(predictions, true_labels):
        confusion[pred, target] += 1

    print(confusion)

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
