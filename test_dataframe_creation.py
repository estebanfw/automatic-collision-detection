from pandas import DataFrame
import create_dataframe

df = create_dataframe.main()

def test():
    """Dataframe building validation
    """
    assert isinstance(df, DataFrame), "Failed to build DataFrame"
    assert df.shape[0] > 100, "Review dataframe rows"
    assert df.shape[1] > 10, "Review dataframe columns"