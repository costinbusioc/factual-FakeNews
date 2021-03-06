import pandas as pd


input_file = "factual_big.csv"
real_label_to_label = {
    #"Adevărat": 0,
    #"Parțial Adevărat": 1,
    "AdevÄƒrat": 0,
    "ParÈ›ial AdevÄƒrat": 1,
    #"Parțial Fals": 2,
    "ParÈ›ial Fals": 2,
    "Fals": 3,
}

labels = []
df = pd.read_csv(input_file)
df = df[
    (
        (df.label != "Imposibil de verificat") &
        #(df.label != "In mandat") &
        (df.label != "Ã�n mandat") &
        (df.label != "Trunchiat") &
        #(df.label != "In afara mandatului") &
        (df.label != "Ã�n afara mandatului") &
        #(df.label != "Numai cu sprijin institutional")
        (df.label != "Numai cu sprijin instituÈ›ional")
    )
]

real_labels = df["label"].tolist()
labels = [real_label_to_label[label] for label in real_labels]

df["label"] = labels

df.to_csv(f"labeled_{input_file}", encoding="utf-8")
