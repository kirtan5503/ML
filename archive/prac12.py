import pandas as pd


def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("C:\Users\kirta\Documents\ML\archive\prices.csv")
    # TODO: Print last 5 rows of the data frame
    df.head(3)


if __name__ == "__main__":
    test_run()
