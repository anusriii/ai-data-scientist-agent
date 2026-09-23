import pandas as pd

from analysis.profiler import generate_profile


def test_numeric_profile():
    df = pd.DataFrame({
        "age": [10, 20, 30, 40, 50]
    })

    profile = generate_profile(df)

    age = profile["columns"]["age"]

    assert age["mean"] == 30
    assert age["median"] == 30
    assert age["min"] == 10
    assert age["max"] == 50