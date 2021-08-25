import pytest
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ArgumentError

from daqm.data.db_table import DBTableData, DBTableQuery
from daqm.data.query import Query
from tests.data.test_query import BaseTestQuery
from tests.utils import TEST_ORDER_DATA



@pytest.mark.run(order=TEST_ORDER_DATA + 11)
class TestDBTableQuery(BaseTestQuery):
  @pytest.fixture(autouse=True)
  def create_data(self, conn):
    super().create_data()
    try:
      self.conn = conn
      self.data = DBTableData.from_df(self.conn, self.df)
      self.data2 = DBTableData.from_df(self.conn, self.df2)
    except (OperationalError, ArgumentError):
      pytest.skip("Connecting Data Test DB Failed. Skipping tests.")
