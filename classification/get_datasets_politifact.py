import pandas as pd
from sklearn.model_selection import train_test_split


files = ["politifact.csv", "politifact_covid.csv"]
folder = "PolitifactDatasets/"

def write_csv(filename, col_names, cols):
    df = pd.DataFrame(cols)
    df = df.transpose()

    with open(filename, 'w', encoding='utf-8', newline = '\n') as f:
        df.to_csv(f, header=col_names)

def get_train_test_files():
    for file in files:
        train_df_list = []
        test_df_list = []
        df = pd.read_csv(file)
        df_label_0 = df[df['label'] == 0]
        df_label_1 = df[df['label'] == 1]
        df_label_2 = df[df['label'] == 2]
        df_label_3 = df[df['label'] == 3]
        df_list = [df_label_0, df_label_1, df_label_2, df_label_3]
        for item in df_list:
            train_df, test_df = train_test_split(item, test_size = 0.25)
            train_df_list.append(train_df)
            test_df_list.append(test_df)

        train_ = pd.concat(train_df_list)
        test_ = pd.concat(test_df_list)

        train_filename = folder + file.split('.')[0] + "_train.csv"
        test_filename = folder + file.split('.')[0] + "_test.csv"

        print (file)
        for label in range(4):
            train = train_[train_['label'] == label].shape[0]
            test = test_[test_['label'] == label].shape[0]
            total_label = train + test
            print ("Label ", label, ": -> train: ", train,
                "(", train * 100 / total_label,"%) ",
                "| -> test: ", test,
                "(", test * 100 / total_label,"%) ",
                "| total: ", total_label)

        write_csv(train_filename, train_.columns.values, [train_[column].to_list() for column in list(train_.columns.values)])
        write_csv(test_filename, test_.columns.values, [test_[column].to_list() for column in list(test_.columns.values)])

def main():
    get_train_test_files()

if __name__ == "__main__":
    main()
