import argparse

import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

train_files = ["FactualDatasets/politifact_factual_train.csv"]
test_files = ["FactualDatasets/factual_test_5.csv"]

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class Model:
    def __init__(self):
        self.train_args = {
            "reprocess_input_data": True,
            "fp16": False,
            "num_train_epochs": 8,
        }

    def generate_df(self, input_file):
        df = pd.read_csv(input_file)
        df["labels"] = df["label"]
        return df[["text", "labels"]]

    def read_df(self, input_file):
        df = pd.read_csv(input_file)
        df["labels"] = df["label"]
        df = df.drop(['Unnamed: 0'], axis=1)
        df = df.drop(['label'], axis=1)
        return df

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

    def test_model(self, test_file, model_type, model_dir, use_cuda=False):
        self.train_args["output_dir"] = out_dir
        test_df = self.generate_df(test_file)
        model = ClassificationModel(
            model_type,
            f"{model_dir}/",
            num_labels=4,
            args=self.train_args,
            use_cuda=use_cuda,
        )
        result, model_outputs, wrong_predictions = model.eval_model(
            test_df,
            f1=self.f1_multiclass,
            acc=accuracy_score,
            precision=self.precision_multiclass,
            recall=self.recall_multiclass,
        )
        print(result)

    def predict(self, test_file, model_type, model_dir, use_cuda=False):
        self.train_args["output_dir"] = out_dir
        test_df = self.read_df(test_file)
        model = ClassificationModel(
            model_type,
            f"{model_dir}/",
            num_labels=4,
            args=self.train_args,
            use_cuda=use_cuda,
        )
        predictions, raw_outputs = model.predict(test_df['text'].to_list())
        test_df['predictions'] = predictions
        test_predictions = f"{test_file[:-4]}_predictions_{model_type}.csv"
        write_csv(test_predictions, list(test_df.columns.values), [test_df[column].to_list() for column in list(test_df.columns.values)])

        print(classification_report(list(test_df["labels"]), predictions, digits=3))


def write_csv(filename, col_names, cols):
    df = pd.DataFrame(cols)
    df = df.transpose()

    with open(filename, 'w', encoding='utf-8', newline = '\n') as f:
            df.to_csv(f, header=col_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="train model")
    parser.add_argument("-e", "--eval", action="store_true", help="eval model")
    parser.add_argument("-p", "--predict", action="store_true", help="eval model")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA for the model")
    parser.add_argument("--model-type", type=str, default="bert")
    parser.add_argument("--model-name", type=str, default="bert-base-multilingual-cased")

    args = parser.parse_args()

    if not args.train and not args.eval and not args.predict:
        parser.print_help()
        exit(0)

    model_type = args.model_type
    model_name = args.model_name

    model = Model()
    if args.train:
        for train_file in train_files:
            print(f"Training {model_type}: {model_name} on {train_file}")
            out_dir = f"{train_file.split('.')[0]}_{model_name}"
            model.train_model(train_file, model_type, model_name, out_dir, args.use_cuda)

    if args.eval:
        for i, train_file in enumerate(train_files):
            print(f"Evaluating {model_type}: {model_name} on {test_files[i]}")
            out_dir = f"{train_file.split('.')[0]}_{model_name}"
            model.test_model(test_files[i], model_type, out_dir, args.use_cuda)

    if args.predict:
        for i, train_file in enumerate(train_files):
            print(f"Predicting {model_type}: {model_name} on {test_files[i]}")
            out_dir = f"{train_file.split('.')[0]}_{model_name}"
            model.predict(test_files[i], model_type, out_dir, args.use_cuda)
