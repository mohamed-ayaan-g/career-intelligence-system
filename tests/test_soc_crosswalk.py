import pandas as pd

from src.data.soc_crosswalk import add_bls_soc_column, onet_soc_to_bls_soc


def test_onet_soc_to_bls_soc_strips_detail_suffix():
    assert onet_soc_to_bls_soc("15-1252.00") == "15-1252"
    assert onet_soc_to_bls_soc("29-1141.01") == "29-1141"


def test_add_bls_soc_column():
    df = pd.DataFrame({"onetsoc_code": ["29-1141.01", "29-1141.02", "29-1141.03"]})
    result = add_bls_soc_column(df)
    assert (result["bls_soc_code"] == "29-1141").all()


def test_many_onet_codes_collapse_to_one_bls_code():
    """The known nursing-specialty example from the module docstring."""
    codes = ["29-1141.01", "29-1141.02", "29-1141.03"]
    bls_codes = {onet_soc_to_bls_soc(c) for c in codes}
    assert bls_codes == {"29-1141"}
