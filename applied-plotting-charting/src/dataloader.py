import pandas as pd
import numpy as np


class Dataloader:
    def __init__(self):
        self.ironman = self._load_ironman()

    def _load_countries(self, aname="data/alpha3.csv", dname="data/distances.csv"):
        alpha = pd.read_csv(aname, names=["Alpha", "Country"]).apply(
            lambda s: s.str.strip()
        )
        dists = pd.read_csv(
            dname,
            sep=";",
            names=["Country", "Distance"],
            usecols=[0, 1],
            converters={
                "Country": lambda s: s.strip(),
                "Distance": lambda n: np.int64(n.replace(",", "").strip()),
            },
        )
        df = pd.merge(alpha, dists, on="Country")
        return pd.concat(
            [
                df,
                pd.DataFrame(
                    {"Alpha": ["CAN"], "Distance": [0], "Country": ["Canada"]}
                ),
            ],
            ignore_index=True,
        )

    def _load_ironman(self, fname="data/ironman2018.csv"):
        df = pd.read_csv(
            fname,
            names=[
                "Overall",
                "GenderRank",
                "DivRank",
                "Name",
                "BIB",
                "AgeGroup",
                "Alpha",
                "Swim",
                "Bike",
                "Run",
                "Total",
            ],
            dtype={"Overall": np.int64,},
            converters={
                k: lambda d: pd.Timedelta(d).seconds
                for k in ["Swim", "Bike", "Run", "Total"]
            },
        )
        df["Transition"] = df["Total"] - df[["Swim", "Bike", "Run"]].sum(axis=1)
        df[["Name", "Alpha", "AgeGroup"]] = df[["Name", "Alpha", "AgeGroup"]].apply(
            lambda s: s.str.strip()
        )
        return pd.merge(df, self._load_countries(), on="Alpha")


if __name__ == "__main__":
    import pdb

    dl = Dataloader()
    bar_data = dl.ironman.groupby("Country").size() / len(dl.ironman) * 100
    bar_data.sort_values(inplace=True, ascending=False)
    bar = bar_data[6:].plot(kind="bar")  # TODO continue
    pdb.set_trace()
