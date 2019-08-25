import pandas as pd

from sklearn.model_selection import KFold

if __name__ == "__main__":
    train_df = pd.read_csv("./crohme-train/train.csv")
    kfold = KFold(n_splits=10, shuffle=True, random_state=1337)
    train_idx, val_idx = list(kfold.split(train_df))[0]
    train_df, val_df = (
        train_df.iloc[train_idx].reset_index(),
        train_df.iloc[val_idx].reset_index(),
    )
    train_df.to_csv("./crohme-train/train.csv")
    val_df.to_csv("./crohme-train/val.csv")
