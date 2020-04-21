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

    def get_top5_countries(self):
        data = self.ironman.groupby("Alpha").size() / len(self.ironman) * 100
        data.sort_values(inplace=True, ascending=False)
        return data[:5].append(pd.Series([data[5:].sum()], index=["Others"]))

    def get_time_means(self):
        cols = ["AgeGroup", "Swim", "Bike", "Run", "Transition", "Total"]
        data = self.ironman[cols].groupby("AgeGroup").apply(np.mean)
        return data.sort_values(by=["Total"]).drop(["Total"], axis=1) / 3600

    def get_dists_quantiles(self):
        data = self.ironman[["AgeGroup", "DivRank", "Distance"]].copy()
        cats = [1, 2, 3, 4]
        data["DivRankQ"] = 0
        for name, group in data.groupby("AgeGroup"):
            quants = pd.cut(group["DivRank"], 4, labels=cats)
            data.loc[quants.index, "DivRankQ"] = quants
        return data

    def get_ttest_data(self):
        data = self.ironman[["AgeGroup", "DivRank", "Alpha"]].copy()
        mask = data["Alpha"] == "CAN"
        can, others = data[mask], data[~mask]
        return can["DivRank"], others["DivRank"]


class TestClass:
    def __init__(self):
        self.val = 42


if __name__ == "__main__":
    dl = Dataloader()
    data_quant = dl.get_dists_quantiles()
    data_ttest = dl.get_ttest_data()
