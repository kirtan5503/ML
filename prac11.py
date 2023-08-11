import pandas as pd


def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("C:/Users/kirta/Documents/ML/archive/prices.csv")
    df.head(4)