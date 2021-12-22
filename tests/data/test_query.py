import pytest
import pandas as pd
import numpy as np

from abc import abstractmethod
from collections import defaultdict
from datetime import date, timedelta

from daqm.data.columns import SimpleColumn, ConstantColumn, FunctionalColumn, OperatedColumn, and_, or_
from daqm.data.query import Query, func
from tests.utils import TEST_ORDER_DATA


TEST_COLUMN_LIST = [
    SimpleColumn("simple", None),
    ConstantColumn(5),
    FunctionalColumn("max", SimpleColumn("simple", None)),
    OperatedColumn(
        SimpleColumn("simple", None),
        ConstantColumn(0.123),
        "add")
]


@pytest.mark.run(order=TEST_ORDER_DATA + 1)
class TestQuery:
  @pytest.fixture(autouse=True)
  def create_query(self):
    self.query = Query(None)

  def test_select(self):
    self.query.select(TEST_COLUMN_LIST[0])
    assert len(self.query.select_list) == 1

    self.query.select(*TEST_COLUMN_LIST[1:])
    assert len(self.query.select_list) == len(TEST_COLUMN_LIST)

  def test_groupby(self):
    self.query.groupby(*TEST_COLUMN_LIST)
    assert len(self.query.groupby_list) == len(TEST_COLUMN_LIST)

  def test_orderby(self):
    self.query.orderby(
        TEST_COLUMN_LIST[0].desc(),
        TEST_COLUMN_LIST[1])

    assert len(self.query.orderby_list) == 2

  def test_where(self):
    self.query.where(
        TEST_COLUMN_LIST[0] == 123,
        TEST_COLUMN_LIST[1] > 123
    )
  # Data 가 있어야함. 추후 테스트 정의한다.
  # def test_join(self):
  #   self.query.join(None, on="test_col", how="inner")
  #   assert len(self.query.join_list) == 1


# @pytest.mark.skip(reason="This is abstract class for query test")
class BaseTestQuery:
  def create_data(self):
    cols = [
        "intA", "intB",
        "floatA", "floatB",
        "stringA", "stringB",
        "dateA", "dateB",
        "dateYear", "dateMonth", "dateDay",
        "null"]
    self.df = pd.DataFrame([
        [123, 321, 0.5342, 0.7895, "string", "string2", "2020-07-15", "2020-07-20", 2020, 7, 15, None],
        [123, 320, 0.5341, 0.7895, "stragdng", "string2", "2020-07-15", "2020-07-23", 1999, 7, 12, None],
        [124, 124, -0.8888, 0.7845, "sadadgng", "string2", "2020-07-15", "2020-07-22", 1969, 12, 13, None]
    ], columns=cols)
    self.df.dateA = pd.to_datetime(self.df.dateA)
    self.df.dateB = pd.to_datetime(self.df.dateB)
    self.col_to_idx = {col: idx for idx, col in enumerate(cols)}

    cols2 = ["intA", "intB", "floatA", "newFloatB"]
    self.df2 = pd.DataFrame([
        [123, 321, 0.123123, 0.124],
        [123, 320, 0.123123, 0.975]
    ], columns=cols2)
    self.col_to_idx2 = {col: idx for idx, col in enumerate(cols)}

  def query_to_df(self, query: Query):
    return query.apply().to_df()

  def test_select(self):
    query = self.data.query.select(
        self.data.c.intA,
        self.data.c.floatA.label("floatChange"),
        self.data.c.floatA + self.data.c.floatB,
        self.data.c.floatA - self.data.c.floatB,
        (self.data.c.floatA * self.data.c.floatB).label("float_mul"),
        (self.data.c.floatA / self.data.c.floatB).label("float_div"),
        (self.data.c.floatA // self.data.c.floatB).label("float_floordiv"),
        (self.data.c.floatA < self.data.c.floatB).label("float_lt"),
        (self.data.c.floatA <= self.data.c.floatB).label("float_le"),
        (self.data.c.floatA == self.data.c.floatB).label("float_eq"),
        (self.data.c.floatA != self.data.c.floatB).label("float_ne"),
        (self.data.c.floatA > self.data.c.floatB).label("float_gt"),
        (self.data.c.floatA >= self.data.c.floatB).label("float_ge"),
        (self.data.c.floatA + self.data.c.floatB * self.data.c.intA).label("multi_op1"),
        (self.data.c.intB + 100).label("operator_with_num1"),
        self.data.c.stringA.notlike("%%s%%").label("notlike_with_percent_sign"),
        self.data.c.stringB.ilike("S______").label("case_insensitive_like"),
        self.data.c.stringA.notilike("S_").label("notilike_with_underscore")
    )
    result_df = self.query_to_df(query)

    assert list(result_df.columns) == [
        "intA",
        "floatChange",
        "(floatA + floatB)",
        "(floatA - floatB)",
        "float_mul",
        "float_div",
        "float_floordiv",
        "float_lt",
        "float_le",
        "float_eq",
        "float_ne",
        "float_gt",
        "float_ge",
        "multi_op1",
        "operator_with_num1",
        "notlike_with_percent_sign",
        "case_insensitive_like",
        "notilike_with_underscore"]

    for source, result in zip(self.data.to_df().values, result_df.values):
      # Check select simple column
      assert result[0] == source[self.col_to_idx["intA"]]
      assert result[1] == source[self.col_to_idx["floatA"]]

      # Check select operator
      assert result[2] == (source[self.col_to_idx["floatA"]] + source[self.col_to_idx["floatB"]])
      assert result[3] == (source[self.col_to_idx["floatA"]] - source[self.col_to_idx["floatB"]])
      assert result[4] == (source[self.col_to_idx["floatA"]] * source[self.col_to_idx["floatB"]])
      assert result[5] == (source[self.col_to_idx["floatA"]] / source[self.col_to_idx["floatB"]])
      assert result[6] == (source[self.col_to_idx["floatA"]] // source[self.col_to_idx["floatB"]])
      assert result[7] == (source[self.col_to_idx["floatA"]] < source[self.col_to_idx["floatB"]])
      assert result[8] == (source[self.col_to_idx["floatA"]] <= source[self.col_to_idx["floatB"]])
      assert result[9] == (source[self.col_to_idx["floatA"]] == source[self.col_to_idx["floatB"]])
      assert result[10] == (source[self.col_to_idx["floatA"]] != source[self.col_to_idx["floatB"]])
      assert result[11] == (source[self.col_to_idx["floatA"]] > source[self.col_to_idx["floatB"]])
      assert result[12] == (source[self.col_to_idx["floatA"]] >= source[self.col_to_idx["floatB"]])

      assert result[-5] == (source[self.col_to_idx["floatA"]]
                            + source[self.col_to_idx["floatB"]]
                            * source[self.col_to_idx["intA"]])

      # Check select operator with numeric
      assert result[-4] == (source[self.col_to_idx["intB"]] + 100)
      assert result[-3] == False
      assert result[-2]
      assert result[-1]

  def test_where(self):
    query = self.data.query.select(
        self.data.c.intA
    ).where(
        self.data.c.intA == 123
    )
    result_df = self.query_to_df(query)

    for result in result_df.values:
      assert result[0] == 123

  def test_where_multi(self):
    query = self.data.query.select(
        self.data.c.intA,
        self.data.c.intB,
        self.data.c.stringA,
        self.data.c.stringB
    ).where(
        self.data.c.intA == 123,
        self.data.c.intB > 320,
        self.data.c.stringA.notlike("%%d%%"),
        self.data.c.stringB.ilike("STRING%%")
    )
    result_df = self.query_to_df(query)

    for result in result_df.values:
      assert result[0] == 123
      assert result[1] > 320
      assert result[2] == "string"
      assert result[3] == "string2"

  @pytest.mark.parametrize("desc", [True, False])
  def test_orderby(self, desc):
    query = self.data.query.select(
        self.data.c.floatA.label("floatA_changed"),
        self.data.c.floatB
    ).orderby(
        self.data.c.floatA.desc() if desc else self.data.c.floatA
    )
    result_df = self.query_to_df(query)

    prev_value = None
    for result in result_df.values:
      cur_value = result[0]
      if prev_value is not None:
        if desc:
          assert prev_value >= cur_value
        else:
          assert prev_value <= cur_value

      prev_value = cur_value

  def test_orderby_operated_column(self):
    query = self.data.query.select(
        self.data.c.floatA,
        self.data.c.floatB
    ).orderby(
        self.data.c.floatA + self.data.c.floatB
    )
    result_df = self.query_to_df(query)

    prev_value = None
    for result in result_df.values:
      cur_value = result[0] + result[1]
      if prev_value is not None:
        assert prev_value <= cur_value

      prev_value = cur_value

  @pytest.mark.parametrize("int_a_where", [
      None,
      123
  ])
  def test_groupby_select_orderby(self, int_a_where):
    query = self.data.query.select(
        self.data.c.intA,
        func.sum(self.data.c.floatA).label("floatA_sum"),
        func.min(self.data.c.floatA).label("floatA_min"),
        func.max(self.data.c.floatA).label("floatA_max"),
        func.avg(self.data.c.floatA).label("floatA_avg"),
        func.mean(self.data.c.floatA).label("floatA_mean")
    ).groupby(
        self.data.c.intA
    ).orderby(
        func.sum(self.data.c.floatA),
    )

    if int_a_where is not None:
      query = query.where(
          self.data.c.intA == int_a_where
      )

    result_df = self.query_to_df(query)

    assert list(result_df.columns) == [
        "intA",
        "floatA_sum",
        "floatA_min",
        "floatA_max",
        "floatA_avg",
        "floatA_mean"
    ]

    grouped_dict = defaultdict(list)
    for row in self.data.to_df().values:
      grouped_dict[row[self.col_to_idx["intA"]]].append(row[self.col_to_idx["floatA"]])

    prev_value = None
    for result in result_df.values:
      values = grouped_dict[result[0]]
      assert np.sum(values) == result[1]
      assert np.min(values) == result[2]
      assert np.max(values) == result[3]
      assert np.mean(values) == result[4]
      assert np.mean(values) == result[5]

      cur_value = result[1]
      if prev_value is not None:
        assert prev_value <= cur_value
      prev_value = cur_value

      if int_a_where is not None:
        assert int_a_where == result[0]

  @pytest.mark.parametrize("join_how", ["inner", "left", "right"])
  def test_join(self, join_how):
    query = self.data.query.join(
        self.data2, on=["intA", "intB"], how=join_how
    ).select(
        self.data.c.intA,
        self.data.c.intB,
        self.data.c.floatA,
        self.data2.c.floatA.label("floatA_2"),
        self.data2.c.newFloatB
    )

    result_df = self.query_to_df(query)

    assert list(result_df.columns) == [
        "intA",
        "intB",
        "floatA",
        "floatA_2",
        "newFloatB"
    ]

    if join_how == "left":
      assert len(result_df) == len(self.data.to_df())
    elif join_how == "right":
      assert len(result_df) == len(self.data2.to_df())

  def test_agg_function(self):
    query = self.data.query.select(
        func.sum(self.data.c.intB),
        func.min(self.data.c.intB),
        func.max(self.data.c.intB),
        func.avg(self.data.c.intB),
        func.mean(self.data.c.intB).label("mean"),
        func.count(self.data.c.intB),
        func.nunique(self.data.c.intB),
        func.unique(self.data.c.intB),
        func.all(self.data.c.intB)
    )

    result_df = self.query_to_df(query)

    assert len(result_df) == 1

    target_col_values = self.data.to_df().intB.values

    result_values = result_df.values[0]
    assert result_values[0] == np.sum(target_col_values)
    assert result_values[1] == np.min(target_col_values)
    assert result_values[2] == np.max(target_col_values)
    assert result_values[3] == np.mean(target_col_values)
    assert result_values[4] == np.mean(target_col_values)
    assert result_values[5] == len(target_col_values)
    assert result_values[6] == len(set(target_col_values))

    ary_values = result_values[7]
    if isinstance(ary_values, str):
      ary_values = list(map(int, ary_values.split(",")))
    assert set(ary_values) == set(target_col_values)

    ary_values = result_values[8]
    if isinstance(ary_values, str):
      ary_values = list(map(int, ary_values.split(",")))
    assert ary_values == list(target_col_values)

  def test_agg_function_error(self):
    with pytest.raises(ValueError, match=r"Column .* must be in group col!!"):
      query = self.data.query.select(
          self.data.c.intA,
          func.sum(self.data.c.intB)
      )
      result_df = self.query_to_df(query)

    with pytest.raises(ValueError, match=r"Column .* must be in group col!!"):
      query = self.data.query.select(
          self.data.c.intA + func.sum(self.data.c.intB)
      )
      result_df = self.query_to_df(query)

    # pass cases
    query = self.data.query.select(
        func.sum(self.data.c.intB) + 123
    )
    result_df = self.query_to_df(query)

    query = self.data.query.select(
        self.data.c.intA,
        func.sum(self.data.c.intB)
    ).groupby(self.data.c.intA)
    result_df = self.query_to_df(query)

  def test_numeric_function(self):
    query = self.data.query.select(
        func.abs(self.data.c.floatA),
        func.abs(self.data.c.floatB)
    )

    result_df = self.query_to_df(query)

    for source, result in zip(self.data.to_df().values, result_df.values):
      float_a = source[self.col_to_idx["floatA"]]
      float_b = source[self.col_to_idx["floatB"]]

      assert abs(float_a) == result[0]
      assert abs(float_b) == result[1]

  def test_date_function(self):
    query = self.data.query.select(
        func.date_diff(self.data.c.dateA, self.data.c.dateB),
        func.date_year(self.data.c.dateA),
        func.date(self.data.c.dateYear, self.data.c.dateMonth, self.data.c.dateDay),
        self.data.c.dateA + func.date_delta(10),
        self.data.c.dateA + func.date_delta(self.data.c.intA)
    )

    result_df = self.query_to_df(query)

    for source, result in zip(self.data.to_df().values, result_df.values):
      date_a = source[self.col_to_idx["dateA"]]
      date_b = source[self.col_to_idx["dateB"]]
      int_a = source[self.col_to_idx["intA"]]

      date_year = source[self.col_to_idx["dateYear"]]
      date_month = source[self.col_to_idx["dateMonth"]]
      date_day = source[self.col_to_idx["dateDay"]]

      assert result[0] == (date_a - date_b).days

      assert result[1] == date_a.year
      assert result[2] == date(date_year, date_month, date_day)

      assert result[3] == date_a + timedelta(days=10)
      assert result[4] == date_a + timedelta(days=int_a)

  def test_rank_function(self):
    # TODO
    pass

  def test_case_function(self):
    query = self.data.query.select(
        func.case(
            (self.data.c.dateYear > 2000, "new"),
            (self.data.c.dateYear > 1980, "old"),
            else_col="ancestor"
        )
    )

    result_df = self.query_to_df(query)

    for source, result in zip(self.data.to_df().values, result_df.values):
      date_year = source[self.col_to_idx["dateYear"]]

      if date_year > 2000:
        assert result[0] == "new"
      elif date_year > 1980:
        assert result[0] == "old"
      else:
        assert result[0] == "ancestor"

  def test_null(self):
    query = self.data.query.select(
        func.coalesce(self.data.c.null, "asd"),
        func.coalesce(self.data.c.null, self.data.c.stringA),
        func.isnull(self.data.c.null),
        func.notnull(self.data.c.null)
    )

    result_df = self.query_to_df(query)

    for source, result in zip(self.data.to_df().values, result_df.values):
      string_a = source[self.col_to_idx["stringA"]]
      assert result[0] == "asd"
      assert result[1] == string_a
      assert result[2]
      assert not result[3]

  def test_in(self):
    check_list_0 = [123, 125]
    check_list_1 = [-0.123, 111]
    query = self.data.query.select(
        func.in_(self.data.c.intA, check_list_0),
        func.in_(self.data.c.intA, check_list_1)
    )

    result_df = self.query_to_df(query)

    for source, result in zip(self.data.to_df().values, result_df.values):
      int_a = source[self.col_to_idx["intA"]]

      assert result[0] == (int_a in check_list_0)
      assert result[1] == (int_a in check_list_1)

  def test_greatest(self):
    query = self.data.query.select(
        func.greatest(
            self.data.c.intA,
            self.data.c.intB,
            self.data.c.floatA,
            self.data.c.floatB
        ).label("greatest_int"),
        func.greatest(
            self.data.c.dateA,
            self.data.c.dateB
        ).label("greatest_date")
    )
    result = self.query_to_df(query)

    assert result.iloc[0, 0] == 321
    assert result.iloc[1, 0] == 320
    assert result.iloc[2, 0] == 124
    assert result.iloc[0, 1] == date(2020, 7, 20)
    assert result.iloc[1, 1] == date(2020, 7, 23)
    assert result.iloc[2, 1] == date(2020, 7, 22)

  def test_least(self):
    query = self.data.query.select(
        func.least(
            self.data.c.intA,
            self.data.c.intB,
            self.data.c.floatA,
            self.data.c.floatB
        ).label("least_value"),
        func.least(
            self.data.c.stringA,
            self.data.c.stringB,
            self.data.c.null
        ).label("least_str_with_null")
    )
    result = self.query_to_df(query)

    assert result.iloc[0, 0] == 0.5342
    assert result.iloc[1, 0] == 0.5341
    assert result.iloc[2, 0] == -0.8888
    assert result.iloc[0, 1] == "string"
    assert result.iloc[1, 1] == "stragdng"
    assert result.iloc[2, 1] == "sadadgng"

  def test_and(self):
    query = self.data.query.select(
        func.case(
            (
                and_(self.data.c.dateYear > 2000, self.data.c.dateMonth >= 1), "new"
            ),
            (
                and_(self.data.c.dateYear > 1980, self.data.c.dateMonth <= 12), "old"
            ),
            else_col="ancestor"
        )
    )
    result = self.query_to_df(query)
    assert result.iloc[0, 0] == "new"
    assert result.iloc[1, 0] == "old"
    assert result.iloc[2, 0] == "ancestor"

  def test_or(self):
    query = self.data.query.select(
        self.data.c.intA
    ).where(
        or_(
            self.data.c.intA == 123,
            self.data.c.intB == 321
        )
    )
    result = self.query_to_df(query)
    assert [row == 123 for row in result["intA"]]

  def test_apply_function(self):
    def apply_func(row):
      new_row = []
      new_row.append(row[0] * 1024)
      if row[4] == "string":
        new_row.append("it's string!!")
      else:
        new_row.append("something else")
      return new_row

    data = self.data.apply_function(apply_func, columns=["test1", "test2"])
    result_df = data.to_df()

    assert list(result_df.columns) == ["test1", "test2"]
    for source, result in zip(self.data.to_df().values, result_df.values):
      assert result[0] == source[0] * 1024
      if source[4] == "string":
        assert result[1] == "it's string!!"
      else:
        assert result[1] == "something else"
