import pandas as pd


input_file = "factual_coronavirus_diac.csv"
real_label_to_label = {
    "Adevărat": 0,
    "Parțial Adevărat": 1,
    "Numai cu sprijin instituțional": 1,
    "Trunchiat": 2,
    "Parțial Fals": 2,
    "În afara mandatului": 3,
    "Fals": 3,
}

labels = []
df = pd.read_csv(input_file)
df = df[(df.real_label != "Imposibil de verificat") & (df.real_label != "în mandat")]

real_labels = df["real_label"].tolist()
labels = [real_label_to_label[label] for label in real_labels]

df["label"] = labels

df.to_csv(f"labeled_{input_file}", encoding="utf-8")
