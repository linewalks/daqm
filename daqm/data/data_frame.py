"""
DataFrame 모듈

pandas DataFrame을 이용한 Data Wrapper, Query
"""


import functools
import pandas as pd
import numpy as np
import warnings
import sqlalchemy

from collections import defaultdict
from datetime import datetime
from typing import List, Callable, Generator

from daqm.data.data import Data
from daqm.data.columns import Column, OperatedColumn, ConstantColumn, SimpleColumn, FunctionalColumn
from daqm.data.query import Query


def convert_multiindex_columns(
    df: pd.DataFrame,
    delimiter: str = "$"
) -> List[str]:
  """
  pandas.DataFrame의 multi index 컬럼을 delimiter를 이용해 string으로 변환합니다.

  :example: ("ABC", "DEF") -> "ABC$DEF"

  :param panas.DataFrame df:
    변환할 컬럼이 있는 DataFrame
  :param str delimiter:
    사용할 구분자.

  :return: 변환된 컬럼 이름 배열
  :rtype: list
  """
  cols = []
  for col in df.columns:
    if isinstance(col, str):
      cols.append(col)
    else:
      cols.append(delimiter.join(col))
  return cols


class DataFrameQuery:
  """
  pandas DataFrame에 Query하기 위한 class

  :param pandas.DataFrame df:
    쿼리할 데이터가 있는 pandas Dataframe
  :param Query query:
    적용할 Query
  """

  @classmethod
  def query(
      cls,
      df: pd.DataFrame,
      query: Query
  ) -> pd.DataFrame:
    """
    쿼리를 적용해 DataFrame으로 리턴

    :param pandas.DataFrame df:
      쿼리할 데이터가 있는 pandas Dataframe
    :param Query query:
      적용할 Query

    :return: 쿼리가 적용된 pandas DataFrame
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(df)
    return cls(df, query).df

  def __init__(self, df: pd.DataFrame, query: Query):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.query = query
    self.df = df

    self._parse_query()

  def _parse_query(self):
    """
    Query Parsing
    """
    self._parse_join()
    self._parse_where()
    if self.query.groupby_list:
      self._parse_groupby()
    else:
      self._parse_normal()

  @staticmethod
  def _check_and_add_column(
      df: pd.DataFrame,
      col: Column,
      child_only: bool = False
  ):
    """
    컬럼 연산을 위해 필요한 값들을 계산해 df에 추가해준다.
    """
    for child_col in col.children:
      DataFrameQuery._check_and_add_column(df, child_col)

    if child_only:
      return

    if isinstance(col, SimpleColumn):
      if col.target_column_name not in df.columns:
        raise ValueError(f"Column {col.target_column_name} not in DataFrame.")
      df.loc[:, col.name] = df[col.target_column_name]
    elif isinstance(col, ConstantColumn):
      df[col.name] = col.value
    elif isinstance(col, FunctionalColumn):
      # NOTE QueryFunction Marker
      # If add new function in QueryFunction, must add it's implementation here.
      res_col = None
      if col.func == "sum":
        res_col = df[col.columns[0].name].sum()
      elif col.func == "min":
        res_col = df[col.columns[0].name].min()
      elif col.func == "max":
        res_col = df[col.columns[0].name].max()
      elif col.func == "avg":
        res_col = df[col.columns[0].name].mean()
      elif col.func == "count":
        res_col = df[col.columns[0].name].count()
      elif col.func == "unique":
        res_col = df[col.columns[0].name].unique()
        if not col.options["to_string"]:
          warnings.warn("DataFrameData doesn't support array. It will be changed to comma separated string")

        if isinstance(res_col, np.ndarray):
          res_col = col.options["string_delimiter"].join(map(str, sorted(res_col)))
          res_col = pd.Series(res_col, index=df.index)
        else:
          res_col = res_col.apply(lambda x: col.options["string_delimiter"].join(map(str, sorted(x))))
      elif col.func == "nunique":
        res_col = df[col.columns[0].name].nunique()
      elif col.func == "all":
        res_col = df[col.columns[0].name].values
        if isinstance(res_col, np.ndarray):
          res_col = col.options["string_delimiter"].join(map(str, res_col))
          res_col = pd.Series(res_col, index=df.index)
        else:
          res_col = res_col.apply(lambda x: col.options["string_delimiter"].join(map(str, x)))
      elif col.func == "abs":
        res_col = df[col.columns[0].name].abs()
      elif col.func == "round":
        res_col = df[col.columns[0].name].round(col.options["decimals"])
      elif col.func == "ceil":
        res_col = np.ceil(df[col.columns[0].name])
      elif col.func == "trunc":
        res_col = np.trunc(df[col.columns[0].name])
      elif col.func == "floor":
        res_col = np.floor(df[col.columns[0].name])
      elif col.func == "rank":
        if col.columns[1] is None:
          res_col = df[col.columns[0].name].rank()
        else:
          res_col = df.groupby(col.columns[1].name)[col.columns[0].name].rank()
      elif col.func == "date_diff":
        res_col = (df[col.columns[0].name] - df[col.columns[1].name]) / np.timedelta64(1, "D")
      elif col.func == "date_year":
        res_col = df[col.columns[0].name].dt.year
      elif col.func == "date":
        year_col = col.columns[0].name
        month_col = col.columns[1].name
        day_col = col.columns[2].name
        res_col = df.apply(lambda x: datetime(
            int(x[year_col]),
            int(x[month_col]),
            int(x[day_col])), axis=1)
      elif col.func == "date_delta":
        value_col = col.columns[0]
        if isinstance(value_col, ConstantColumn):
          res_col = pd.to_timedelta(value_col.value, unit="days")
        else:
          res_col = pd.to_timedelta(df[value_col.name], unit="days")
      elif col.func == "case":
        res_col = pd.Series(None, index=df.index)
        for idx in range(0, len(col.columns), 2):
          if idx == len(col.columns) - 1:
            # It's else col
            value_col = col.columns[idx]
            if isinstance(value_col, ConstantColumn):
              each_res_col = pd.Series(value_col.value, index=df.index)
            else:
              each_res_col = df[value_col.name]
          else:
            condition_col = col.columns[idx]
            value_col = col.columns[idx + 1]

            def apply_condition(x):
              """조건 적용"""
              if x[condition_col.name]:
                if isinstance(value_col, ConstantColumn):
                  return value_col.value
                else:
                  return x[value_col.name]
              else:
                return None

            each_res_col = df.apply(apply_condition, axis=1)
            if each_res_col.empty:
              each_res_col = pd.Series(None, dtype="float64")
          res_col = res_col.fillna(each_res_col)
      elif col.func == "coalesce":
        res_col = pd.Series(None, index=df.index)
        for each_col in col.columns:
          if isinstance(each_col, ConstantColumn):
            each_res_col = pd.Series(each_col.value, index=df.index)
          else:
            each_res_col = df[each_col.name]
          res_col = res_col.fillna(each_res_col)
      elif col.func == "isnull":
        res_col = pd.isnull(df[col.columns[0].name])
      elif col.func == "notnull":
        res_col = pd.notnull(df[col.columns[0].name])
      elif col.func in ("in", "notin"):
        in_cols = [
            in_col.value if isinstance(in_col, ConstantColumn) else col.name
            for in_col in col.columns[1:]
        ]
        res_col = df[col.columns[0].name].isin(in_cols)
        if col.func == "notin":
          res_col = ~res_col
      elif col.func == "greatest":
        col_list = [each_col.name for each_col in col.columns]
        res_col = df[col_list].apply(
            lambda x: x.max()
            if x.isna() is False
            else x.dropna().max(),
            axis=1
        )
      elif col.func == "least":
        col_list = [each_col.name for each_col in col.columns]
        res_col = df[col_list].apply(
            lambda x: x.min()
            if x.isna() is False
            else x.dropna().min(),
            axis=1
        )
      elif col.func == "and":
        condition_df_list = [df[each_col.name] for each_col in col.columns]
        res_col = functools.reduce(np.logical_and, condition_df_list)
      elif col.func == "or":
        condition_df_list = [df[each_col.name] for each_col in col.columns]
        res_col = functools.reduce(np.logical_or, condition_df_list)
      elif col.func == "cast":
        # TODO: numeric or decimal: 사용자 지정 정밀도 유형 추가
        # TODO: interval 추가(입력되는 unit에 따라 to_timedelta 함수의 unit 파라미터 변경)
        convert_map_dict = {
            "bigint": ["float", "Int64"],
            "int": ["float", "Int32"],
            "smallint": ["float", "Int16"],
            "boolean": ["boolean"],
            "double precision": ["float64"],
            "float": ["float64"],
            "real": ["float32"],
            "date": ["datetime64[ns]"],
            "datetime": ["datetime64[ns]"],
            "time": ["datetime64[ns]"],
            "char": [str],
            "varchar": [str],
            "text": [str]
        }
        target_type_list = convert_map_dict.get(col.options["target_type"])

        res_col = df[col.columns[0].name]
        for target_type in target_type_list:
          res_col = res_col.astype(target_type)

        if col.options["target_type"] == "date":
          res_col = res_col.dt.date
        elif col.options["target_type"] == "time":
          res_col = res_col.dt.time
      else:
        raise NotImplementedError(f"Function {col.func} not implemented for DataFrame.")
      df.loc[:, col.name] = res_col
    elif isinstance(col, OperatedColumn):
      res_col = None
      left_col = col.l_column.value if col.l_column.is_constant() else df[col.l_column.name]
      right_col = col.r_column.value if col.r_column.is_constant() else df[col.r_column.name]

      # NOTE ColumnOperator Marker
      # If add new function in QueryFunction, must add it's implementation here.
      if col.operator == "add":
        res_col = left_col + right_col
      elif col.operator == "sub":
        res_col = left_col - right_col
      elif col.operator == "mul":
        res_col = left_col * right_col
      elif col.operator == "div":
        res_col = left_col / right_col
      elif col.operator == "floordiv":
        res_col = left_col // right_col
      elif col.operator == "lt":
        res_col = left_col < right_col
      elif col.operator == "le":
        res_col = left_col <= right_col
      elif col.operator == "eq":
        res_col = left_col == right_col
      elif col.operator == "ne":
        res_col = left_col != right_col
      elif col.operator == "gt":
        res_col = left_col > right_col
      elif col.operator == "ge":
        res_col = left_col >= right_col
      elif col.operator in ("like", "ilike", "notlike", "notilike"):
        right_col = "^" + right_col + "$"
        right_col = right_col.replace("%%", r"(.|\s)*")
        right_col = right_col.replace("_", r"(.|\s)")
        case = col.operator in ("like", "notlike")
        if col.operator in ("notlike", "notilike"):
          right_col = rf"(?!{right_col})"
        res_col = left_col.str.match(right_col, case=case)
      df.loc[:, col.name] = res_col

  def _parse_where(self):
    """
    where문 Parsing
    """
    for col in self.query.where_list:
      self._check_and_add_column(self.df, col)
      self.df = self.df[self.df[col.name]]

  def _parse_join(self):
    """
    join문 Parsing
    """
    for data, left_on, right_on, how, suffixes in self.query.join_list:
      right_df = data.to_df()
      left_on_names = []
      right_on_names = []

      for col in left_on:
        self._check_and_add_column(self.df, col)
        left_on_names.append(col.name)
      for col in right_on:
        self._check_and_add_column(right_df, col)
        right_on_names.append(col.name)

      self.df = pd.merge(
          self.df,
          right_df,
          left_on=left_on_names,
          right_on=right_on_names,
          how=how,
          suffixes=suffixes
      )

  def _is_array_agg(self, agg_type):
    """
    입력된 agg_type이 배열을 리턴하는지 여부
    """
    return agg_type in ["unique", "all"]

  def _parse_groupby_agg(self, col: Column) -> (str, str, str, str):
    """
    groupby 컬럼 Parsing

    :return:
      컬럼 이름, Aggregation 타입, Aggregate적용 후 컬럼 이름, 변경할 컬럼 이름
    """
    # TODO
    # There are functions or operations that are not allowed in select statement with groupby.
    # Need to raise Exception.
    is_functional = isinstance(col, FunctionalColumn)
    self._check_and_add_column(self.df, col, child_only=is_functional)

    agg_col_name = col.name
    agg_type = agg_type_col_name = "first"
    rename_key = None
    rename_value = None
    if is_functional:
      # NOTE QueryFunction Marker
      # If add new function in QueryFunction, must add it's implementation here.
      # Here is for group by agg functions. Might not be needed.
      agg_type = agg_type_col_name = col.func
      if agg_type == "avg":
        agg_type = agg_type_col_name = "mean"
      elif agg_type == "nunique":
        agg_type = pd.Series.nunique
      elif self._is_array_agg(agg_type):
        if not col.options["to_string"]:
          warnings.warn("DataFrameData doesn't support array. It will be changed to comma separated string")

        if agg_type == "all":
          agg_type = list
          agg_type_col_name = "list"

      agg_col_name = col.columns[0].name

    rename_key = "$".join((agg_col_name, agg_type_col_name))
    rename_value = col.name

    return (agg_col_name, agg_type, rename_key, rename_value)

  def _parse_groupby(self):
    """
    groupby가 있는 쿼리 Parsing
    """
    groupby_column_names = []
    for col in self.query.groupby_list:
      self._check_and_add_column(self.df, col)
      groupby_column_names.append(col.name)

    select_column_names = []
    agg_col_dict = defaultdict(set)
    rename_dict = defaultdict(list)
    array_cols = []              # must change array to string if unique in agg_type
    for col in self.query.select_list:
      if col.name in select_column_names:
        raise ValueError("Duplicate column in select clause. Use label('new_name') to avoid ambiguity.")
      select_column_names.append(col.name)
      if col in self.query.groupby_set:
        continue

      agg_col_name, agg_type, rename_key, rename_value = self._parse_groupby_agg(col)
      agg_col_dict[agg_col_name].add(agg_type)
      if rename_key:
        rename_dict[rename_key].append(rename_value)
        if self._is_array_agg(agg_type):
          array_cols.append((rename_key, col.options["string_delimiter"]))

    orderby_column_names = []
    orderby_ascendings = []
    for col in self.query.orderby_list:
      orderby_column_names.append(col.name)
      orderby_ascendings.append(not col.is_desc())
      if col in self.query.groupby_set:
        continue

      agg_col_name, agg_type, rename_key, rename_value = self._parse_groupby_agg(col)
      agg_col_dict[agg_col_name].add(agg_type)
      if rename_key:
        rename_dict[rename_key].append(rename_value)
        if self._is_array_agg(agg_type):
          array_cols.append((rename_key, col.options["string_delimiter"]))

    self.df = self.df.groupby(groupby_column_names).agg(agg_col_dict)
    self.df.columns = convert_multiindex_columns(self.df)
    for col, delimiter in array_cols:
      self.df.loc[:, col] = self.df[col].apply(lambda x: delimiter.join(map(str, sorted(x))))

    for rename_key in rename_dict:
      for new_col_name in rename_dict[rename_key]:
        self.df.loc[:, new_col_name] = self.df[rename_key]

    if len(self.df) == 0:
      # when data frame is empty, reset_index() does't reset groupby columns
      self.df.reset_index(drop=True)
      for col in groupby_column_names:
        self.df[col] = pd.Series(None, self.df.index)
    else:
      self.df = self.df.reset_index()

    if orderby_column_names:
      self.df = self.df.sort_values(orderby_column_names, ascending=orderby_ascendings)

    if select_column_names:
      self.df = self.df[select_column_names]

  def _parse_normal(self):
    """
    groupby가 없는 일반 쿼리 Parsing
    """
    select_column_names = []
    for col in self.query.select_list:
      self._check_and_add_column(self.df, col)
      if col.name in select_column_names:
        raise ValueError("Duplicate column in select clause. Use label(\"new_name\") to avoid ambiguity.")
      select_column_names.append(col.name)

    orderby_column_names = []
    orderby_ascendings = []
    for col in self.query.orderby_list:
      self._check_and_add_column(self.df, col)
      orderby_column_names.append(col.name)
      orderby_ascendings.append(not col.is_desc())

    if orderby_column_names:
      self.df = self.df.sort_values(orderby_column_names, ascending=orderby_ascendings)

    if select_column_names:
      self.df = self.df[select_column_names]

    # agg 함수라면 drop_duplicates해주어야 1줄로 간다.
    if self.query.is_agg:
      self.df = self.df.drop_duplicates()


class DataFrameData(Data):
  """
  pandas.DataFrame Wrapper

  :param pandas.DataFrame data_df:
    pandas DataFrame 오브젝트
  """

  @classmethod
  def from_sql(cls, conn: sqlalchemy.engine.Connection, sql: str) -> "DataFrameData":
    """
    DB 쿼리 결과를 DataFrameData로 만듭니다

    :param conn:
      DB Connection
    :param str sql:
      SQL문

    :return: 쿼리 결과를 담은 DataFrameData
    :rtype: DataFrameData
    """
    return cls(pd.read_sql(sql, conn))

  def __init__(self, data_df: pd.DataFrame):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.data_df = data_df
    super().__init__(self.data_df.columns)

  def apply_query(self, query: Query) -> "DataFrameData":
    """
    Query를 적용합니다.

    :param Query query:
      적용할 Query Object

    :return: Query가 적용된 Data
    :rtype: DataFrameData
    """
    return DataFrameData(DataFrameQuery.query(self.data_df.copy(), query))

  def to_df(self) -> pd.DataFrame:
    """
    Data를 pandas DataFrame으로 변환합니다.

    :return: pandas.DataFrame 형태로 변환된 데이터.
    :rtype: pandas.DataFrame
    """
    return self.data_df

  def rename_column(self, rename_dict: dict) -> "DataFrameData":
    """
    컬럼의 이름을 재정의합니다.

    :param dict rename_dict:
      기존 컬럼 이름을 key, 변경할 컬럼 이름을 value로 하는 dict

    :return: 컬럼 이름이 변경된 Data
    :rtype: DataFrameData
    """
    data_df = self.data_df.rename(rename_dict, axis="columns")
    return DataFrameData(data_df)

  def copy(self) -> "DataFrameData":
    """
    현재 Data를 복사합니다.

    :return: 복사된 Data
    :rtype: DataFrameData
    """
    return DataFrameData(pd.DataFrame(self.data_df))

  def apply_function(self,
                     func: Callable[[List], List],
                     columns: List[str] = None):
    """
    데이터에 함수를 적용합니다.

    :param Callable func:
      데이터 한 줄을 list로 입력 받아 변환된 줄을 list로 리턴하는 함수.
    :param list columns:
      적용 후 컬럼 이름.

    :return: 함수가 적용된 Data
    :rtype: DataFrameData
    """
    data_df = self.data_df.apply(lambda x: pd.Series(func(x)), axis=1)
    if columns is None:
      data_df.columns = [str(i) for i in range(len(data_df.columns))]
    else:
      data_df.columns = columns
    return DataFrameData(data_df)

  def iter(self) -> Generator[list, None, None]:
    """
    데이터를 한 줄씩 탐색합니다.

    :return: 한 줄씩 list로 리턴하는 Generator.
    :rtype: generator
    """
    for row in self.data_df.values:
      yield row.tolist()

  def count(self) -> int:
    """
    데이터의 row 수를 가져옵니다.

    :return: 데이터 총 row 수
    :rtype: int
    """
    return len(self.data_df)
