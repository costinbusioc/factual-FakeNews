import argparse

import os
import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

train_file = "../helpers/politifact.csv"
test_directory = "splitted_datasets"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_scores(y_test, predictions):
    cnf_matrix = confusion_matrix(y_test, predictions)
    fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    tp = np.diag(cnf_matrix)
    tn = cnf_matrix.sum() - (fp + fn + tp)

    print(f"tp: {tp}")
    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    print('Accuracy: ', accuracy_score(y_test, predictions))


class Model:
    def __init__(self, test_file):
        self.train_args = {
            "reprocess_input_data": True,
            "fp16": False,
            "num_train_epochs": 3,
        }
        self.test_file = test_file

    def generate_df(self, input_file, cols=["text", "labels"], en=False):
        df = pd.read_csv(input_file)
        df["labels"] = df["label"]
        if en:
            df["text"] = df["English"]
        return df[cols]

    def read_df(self, input_file):
        df = pd.read_csv(input_file)
        df["labels"] = df["label"]
        return df

    def compute_result(self, n_folds=10, use_cuda=True):
        politifact_df = self.generate_df(train_file)
        test_df = self.generate_df(self.test_file, en=True)

        folds_dfs = np.array_split(test_df, n_folds)
        for index, test_df in enumerate(folds_dfs):
            print(f"Fold {index}")
            train_df = pd.concat([politifact_df] + folds_dfs[0:index] + folds_dfs[(index + 1):n_folds])
            out_dir = str(index) + "_" + self.test_file

            model = ClassificationModel(
                model_type,
                model_name,
                num_labels=4,
                args=self.train_args,
                use_cuda=use_cuda,
            )

            model.train_model(
                train_df,
                output_dir=out_dir,
            )

            predictions, raw_outputs = model.predict(test_df['text'].to_list())
            get_scores(test_df["labels"], predictions)
            print(classification_report(list(test_df["labels"]), predictions, digits=3))


    def f1_multiclass(self, labels, preds):
        return f1_score(labels, preds, average="micro")

    def precision_multiclass(self, labels, preds):
        return precision_score(labels, preds, average="micro")

    def recall_multiclass(self, labels, preds):
        return recall_score(labels, preds, average="micro")

    def train_model(self, train_file, model_type, model_name, out_dir=None, use_cuda=False):
        self.train_args["output_dir"] = out_dir
        train_df = self.generate_df(train_file)
        model = ClassificationModel(
            model_type,
            model_name,
            num_labels=4,
            args=self.train_args,
            use_cuda=use_cuda,
        )

        model.train_model(
            train_df,
            output_dir=out_dir,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA for the model")
    parser.add_argument("--model-type", type=str, default="bert")
    parser.add_argument("--model-name", type=str, default="bert-base-cased")

    args = parser.parse_args()

    model_type = args.model_type
    model_name = args.model_name

    for subdir, dirs, files in os.walk(test_directory):
        for dir in dirs:
            col_directory = os.path.join(test_directory, dir)
            for col_subdirs, col_dirs, col_files in os.walk(col_directory):
                for file in col_files:
                    print(f"Training {model_type}: {model_name} on {file}")
                    model = Model(os.path.join(col_directory, file))
                    model.compute_result(n_folds=10, use_cuda=args.use_cuda)
                    print("====")
