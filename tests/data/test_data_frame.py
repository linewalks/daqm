import pytest
import pandas as pd

from daqm.data.data_frame import DataFrameData, DataFrameQuery
from daqm.data.query import Query
from tests.data.test_query import BaseTestQuery
from tests.utils import TEST_ORDER_DATA


@pytest.mark.run(order=TEST_ORDER_DATA + 11)
class TestDataFrameQuery(BaseTestQuery):
  @pytest.fixture(autouse=True)
  def create_data(self):
    super().create_data()
    self.data = DataFrameData(self.df)
    self.data2 = DataFrameData(self.df2)
