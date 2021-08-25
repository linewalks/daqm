import pytest
import pandas as pd

from daqm.data import DataFrameData, DBTableData
from daqm.data.db_table import execute_query
from tests.utils import TEST_ORDER_DATA


required_cols = [
    "patient_id",
    "visit_id",
    "drug_start_date",
    "drug_end_date",
    "drug_id"
]
optional_cols = [
    "qty",
    "amt"
]


data_df = pd.DataFrame([], columns=required_cols + optional_cols)

table_name = "temp_test_data_table"
table_query = f"""
    drop table if exists pg_temp.{table_name};
    create temp table {table_name}
    (
      patient_id int,
      visit_id int,
      drug_start_date date,
      drug_end_date date,
      drug_id int,
      qty float,
      amt float
    )
"""


@pytest.mark.run(order=TEST_ORDER_DATA)
@pytest.mark.parametrize("data_type", ["df", "db"])
class TestDataFunctions:
  @pytest.fixture(autouse=True)
  def create_data(self, data_type, conn):

    if data_type == "df":
      self.data = DataFrameData(data_df)
    elif data_type == "db":
      try:
        execute_query(conn, table_query)
        self.data = DBTableData(conn, table_name)
      except Exception as e:
        pytest.skip(f"Connecting Data Test DB Failed. Skipping tests. {e}")

  def test_check_data_columns_missing(self):
    with pytest.raises(ValueError, match=r"Required columns .* missing."):
      self.data.check_data_columns(required_cols + ["dummy"], optional_cols)

  def test_check_data_columns_match(self):
    cols = self.data.check_data_columns(required_cols, optional_cols)
    assert len(cols) == len(required_cols + optional_cols)

  def test_check_data_columns_less_optional_cols(self):
    cols = self.data.check_data_columns(required_cols, optional_cols + ["dummy"])
    assert len(cols) == len(required_cols + optional_cols)

  def test_check_data_columns_additional_cols(self):
    with pytest.warns(UserWarning, match=r"Not supported columns .*"):
      cols = self.data.check_data_columns(required_cols, optional_cols[:-1])
      assert len(cols) == len(required_cols + optional_cols[:-1])
