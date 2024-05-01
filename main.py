"""Import packages"""

import create_dataframe
import linear_regression


def main():
    """Run full algorithm"""
    df = create_dataframe.main()
    lr = linear_regression.training(df)
    test = lr[1]
    model = lr[2]
    return linear_regression.testing(model, test)


if __name__ == "__main__":
    main()
