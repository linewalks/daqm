"""
Query 모듈

ORM(Object Relation Mapping)을 지원합니다.
"""

from typing import Union, List, Optional, Tuple, Type
from collections.abc import Iterable

from daqm.data.columns import Column, FunctionalColumn, SimpleColumn
from daqm.utils.type_convert import elem_to_list


class Query:
  """
  Query class

  :param Data data: 쿼리를 실행할 Base 데이터
  """
  def __init__(
      self,
      data: "Data"):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.data = data
    self.select_list = []
    self.join_list = []
    self.where_list = []
    self.groupby_list = []
    self.groupby_set = set()
    self.orderby_list = []

  def select(self, *cols: Union[Column, List[Column]]):
    """
    컬럼을 선택합니다. sql의 select문 역할입니다.
    Fluent Interface로 구현되었습니다.

    :type cols: Column or list of Column
    :param cols:
      select 컬럼

    :return: self
    """
    new_cols = []
    for col in cols:
      if isinstance(col, Iterable):
        new_cols.extend(col)
      else:
        new_cols.append(col)
    self.select_list.extend([Column.cast(col) for col in new_cols])
    return self

  def groupby(self, *cols: Column):
    """
    그룹 지을 컬럼을 선택합니다. sql의 group by문 역할입니다.
    Fluent Interface로 구현되었습니다.

    :type cols: Column
    :param cols:
      select 컬럼

    :return: self
    """
    self.groupby_list = [Column.cast(col) for col in cols]
    self.groupby_set = set(self.groupby_list)
    return self

  def orderby(self, *cols: Column):
    """
    정렬할 선택합니다. sql의 order by문 역할입니다.
    Fluent Interface로 구현되었습니다.

    :type cols: Column
    :param cols:
      select 컬럼

    :return: self
    """
    self.orderby_list = [Column.cast(col) for col in cols]
    return self

  def join(
      self,
      data: "Data",
      on: Union[List[str], str] = None,
      left_on: Union[List[Column], Column] = None,
      right_on: Union[List[Column], Column] = None,
      suffixes: Tuple[str, str] = ("", "_r"),
      how: str = "inner"
  ):
    """
    데이터를 합칩니다. sql의 join문 역할입니다.
    Fluent Interface로 구현되었습니다.

    :param Data data: 합칠 데이터
    :type on: str or list of str
    :param on: join할 key 컬럼 이름

    :type left_on: Column or list of Column
    :param left_on:
      기존 데이터의 join key 컬럼. left_on 설정시 right_on도 함께 설정되어야합니다.
      left_on과 right_on의 길이는 같아야하며, 같은 순서의 컬럼끼리 매칭됩니다.
    :type right_on: Column or list of Column
    :param right_on:
      합쳐질 데이터의 join key 컬럼. right_on 설정시 left_on도 함께 설정되어야합니다.
      left_on과 right_on의 길이는 같아야하며, 같은 순서의 컬럼끼리 매칭됩니다.
    :type suffixes: tuple(str, str)
    :param suffixes:
      동일한 컬럼명이 있을 시, 컬럼명 뒤에 붙을 텍스트.
      (기존 데이터, 합쳐질 데이터)
    :param str how:
      join 방법. inner, left, right, outer 중 하나.

    :return: self
    """

    if self.data == data:
      data = data.copy()

    if on:
      on = elem_to_list(on)
      left_on = []
      right_on = []
      for col_name in on:
        if col_name not in self.data.columns:
          raise ValueError(f"Left data has no column {col_name}.")
        if col_name not in data.columns:
          raise ValueError(f"Right data has no column {col_name}.")

        left_on.append(self.data.c[col_name])
        right_on.append(data.c[col_name])
    else:
      left_on = elem_to_list(left_on)
      right_on = elem_to_list(right_on)
    self.join_list.append((data, left_on, right_on, how, suffixes))
    return self

  def where(self, *condition_args: Column):
    """
    조건문을 추가합니다. sql의 where문 역할입니다.

    Fluent Interface로 구현되었습니다.

    :type condition_args: Column
    :param condition_args:
      조건문 컬럼

    :return: self
    """
    self.where_list.extend(condition_args)
    return self

  def _find_columns_in_children(
      self,
      col: Column,
      target_cls: Type[Column] = SimpleColumn
  ) -> List[Column]:
    """
    입력된 컬럼 col의 자식 컬럼 중, target_cls를 모두 리턴합니다.
    """
    result_list = []
    if isinstance(col, target_cls):
      result_list.append(col)
      
    for child in col.children:
      result_list.extend(self._find_columns_in_children(child, target_cls))
    return result_list

  def _find_columns_need_agg(
      self,
      col: Column,
      in_agg=False
  ) -> Tuple[List[Column], List[Column]]:
    """
    입력된 컬럼 col의 자식 컬럼 중,
    aggregate 함수에 들어가 있지 않은 SimpleColumn과
    aggregate 함수에 들어가 있는 SimpleColumn을 리턴합ㄴ다.
    """
    in_agg_list = []
    not_in_agg_list = []
    if isinstance(col, FunctionalColumn):
      if col.is_agg:
        in_agg = True
    elif isinstance(col, SimpleColumn):
      if in_agg:
        in_agg_list.append(col)
      else:
        not_in_agg_list.append(col)

    for child in col.children:
      child_in_agg, child_not_in_agg = self._find_columns_need_agg(child, in_agg)
      in_agg_list.extend(child_in_agg)
      not_in_agg_list.extend(child_not_in_agg)

    return in_agg_list, not_in_agg_list

  def _check_agg(self) -> bool:
    """
    Agg 함수를 체크하여, 필요한 컬럼이 groupby에 들어가 있는지 확인하고 없으면 에러를 냅니다.
    agg 함수가 있는지 여부를 리턴합니다.
    """

    group_simple_col_set = set()
    for col in self.groupby_list:
      group_simple_col_set.update(self._find_columns_in_children(col))

    in_agg_list = []
    not_in_agg_list = []
    for col in self.select_list:
      col_in_agg, col_not_in_agg = self._find_columns_need_agg(col)
    
      in_agg_list.extend(col_in_agg)
      not_in_agg_list.extend(col_not_in_agg)    

    # 둘 다 있을때만 문제
    if in_agg_list and not_in_agg_list:
      for col in not_in_agg_list:
        if col not in group_simple_col_set:
          raise ValueError(f"Column {col.name} must be in group col!!")
    
    return len(in_agg_list) > 0

  def apply(self) -> "Data":
    """
    데이터에 쿼리를 적용합니다.

    :return: 쿼리가 적용된 데이터
    :rtype: Data
    """
    self.is_agg = self._check_agg()
    return self.data.apply_query(self)


class QueryFunction:
  """
  쿼리에 사용될 함수입니다.

  :example: 다음과 같이 사용합니다.

    .. code-block:: python

      from daqm.data.query import func

      func.sum(data.c.col1)
      func.mean(data.c.col2)

  """
  # !!! IMPORTANT NOTE      QueryFunction Marker
  # If add function here, need to implement it's functionality.
  # Search for "QueryFunction Marker" comments

  # Aggregate Functions
  @staticmethod
  def sum(col: Column) -> FunctionalColumn:
    """
    합계 sum
    """
    return FunctionalColumn("sum", col, is_agg=True)

  @staticmethod
  def min(col: Column) -> FunctionalColumn:
    """
    최소값 min
    """
    return FunctionalColumn("min", col, is_agg=True)

  @staticmethod
  def max(col: Column) -> FunctionalColumn:
    """
    최대값 max
    """
    return FunctionalColumn("max", col, is_agg=True)

  @staticmethod
  def avg(col: Column) -> FunctionalColumn:
    """
    평균 avg
    """
    return FunctionalColumn("avg", col, is_agg=True)

  @staticmethod
  def mean(col: Column) -> FunctionalColumn:
    """
    평균 mean
    func.avg 와 같음.(avg로 변환)
    """
    return QueryFunction.avg(col)

  @staticmethod
  def count(col: Column) -> FunctionalColumn:
    """
    수 count
    """
    return FunctionalColumn("count", col, is_agg=True)

  @staticmethod
  def unique(col, to_string: bool = False, string_delimiter: str = ",") -> FunctionalColumn:
    """
    유니크한 값 unique, string_agg distinct, array_agg distinct

    :param bool to_string:
      False이면 Array 형태로.
      True이면 string_delimiter를 이용해 string 형태로 변환.
    :param str string_delimiter:
      to_string이 True일 때 사용. 값 사이 구분자.
    """
    return FunctionalColumn(
        "unique",
        col,
        to_string=False,
        string_delimiter=string_delimiter,
        is_agg=True
    )

  @staticmethod
  def nunique(col: Column) -> FunctionalColumn:
    """
    유니크한 값의 수 nunique, count distinct
    """
    return FunctionalColumn("nunique", col, is_agg=True)

  @staticmethod
  def all(col: Column, to_string=False, string_delimiter=",") -> FunctionalColumn:
    """
    모든 값 all, string_agg, array_agg

    :param bool to_string:
      False이면 Array 형태로.
      True이면 string_delimiter를 이용해 string 형태로 변환.
    :param str string_delimiter:
      to_string이 True일 때 사용. 값 사이 구분자.
    """
    return FunctionalColumn(
        "all",
        col,
        to_string=False,
        string_delimiter=string_delimiter,
        is_agg=True)

  # Numeric Functions
  @staticmethod
  def abs(col: Column) -> FunctionalColumn:
    """
    절대값 abs
    """
    return FunctionalColumn("abs", col)

  @staticmethod
  def round(col: Column, decimals: int = 0) -> FunctionalColumn:
    """
    반올림 round
    """
    return FunctionalColumn("round", col, decimals=decimals)

  @staticmethod
  def ceil(col: Column) -> FunctionalColumn:
    """
    올림 nearest integer greater than or equal to argument
    """
    return FunctionalColumn("ceil", col)

  @staticmethod
  def trunc(col: Column) -> FunctionalColumn:
    """
    내림 truncate toward zero
    """
    return FunctionalColumn("trunc", col)

  @staticmethod
  def floor(col: Column) -> FunctionalColumn:
    """
    내림 nearest integer less than or equal to argument
    """
    return FunctionalColumn("floor", col)

  # Date Functions
  @staticmethod
  def date_diff(end_date_col: Column, start_date_col: Column) -> FunctionalColumn:
    """
    날짜 차이

    :param end_date_col: 종료일
    :param start_date_col: 시작일
    """
    return FunctionalColumn("date_diff", end_date_col, start_date_col)

  @staticmethod
  def date_year(date_col: Column) -> FunctionalColumn:
    """
    연도. 날짜에서 연도를 가져옵니다.
    dt.year, extract(year from )
    """
    return FunctionalColumn("date_year", date_col)

  @staticmethod
  def date(year_col: Column, month_col: Column, day_col: Column) -> FunctionalColumn:
    """
    날짜를 설정합니다.

    :param year_col: 연도 컬럼
    :param month_col: 월 컬럼
    :param day_col: 일 컬럼
    """
    return FunctionalColumn("date", year_col, month_col, day_col)

  @staticmethod
  def date_delta(col: Column) -> FunctionalColumn:
    """
    숫자를 날짜 차이 (time delta)로 변경합니다.

    to_timedelta, N days as interval
    """
    col = Column.cast(col)
    return FunctionalColumn("date_delta", col)

  # Rank Functions
  @staticmethod
  def rank(
      rank_col: Column,
      partition_col: Optional[Column] = None
  ) -> FunctionalColumn:
    """
    순위를 매깁니다.

    :param rank_col:
      순위를 매길 컬럼
    :param partition_col:
      그룹지을 컬럼
    """
    return FunctionalColumn("rank", rank_col, partition_col)

  # Case Functions
  @staticmethod
  def case(
      *condition_args: Tuple[Column, Union[Column, int, float]],
      else_col: Union[Column, int, float, str] = None
  ) -> FunctionalColumn:
    """
    조건에 따라 값을 설정합니다.

    :param args:
      조건 컬럼과 결과 값의 Tuple.
      tuple of condition column and value column
    :param else_col: else value column
    """
    columns = []
    for condition_col, value_col in condition_args:
      columns.append(condition_col)
      columns.append(value_col)
    if else_col is not None:
      columns.append(Column.cast(else_col))
    return FunctionalColumn("case", *columns)

  @staticmethod
  def greatest(*cols: Union[Column, int, float, str]) -> FunctionalColumn:
    return FunctionalColumn("greatest", *cols)

  @staticmethod
  def least(*cols: Union[Column, int, float, str]) -> FunctionalColumn:
    return FunctionalColumn("least", *cols)

  # NULL
  @staticmethod
  def coalesce(*cols: Union[Column, int, float]) -> FunctionalColumn:
    """
    값이 null인 경우 대체값을 설정합니다.
    """
    return FunctionalColumn("coalesce", *cols)

  @staticmethod
  def isnull(col: Column) -> FunctionalColumn:
    """
    값이 null이면 true를 반환합니다.
    """
    return FunctionalColumn("isnull", col)

  @staticmethod
  def notnull(col: Column) -> FunctionalColumn:
    """
    값이 null이 아니면 true를 반환합니다.
    """
    return FunctionalColumn("notnull", col)

  # In
  @staticmethod
  def in_(target_col: Column, in_cols: List[Union[int, float, str]]):
    """
    컬럼의 값이 입력된 리스트 중 있는지 확인합니다. isin, in

    :param target_col: 확인하고 싶은 컬럼
    :param in_cols: 값 리스트
    """
    in_cols = [Column.cast(col) for col in in_cols]
    return FunctionalColumn("in", target_col, *in_cols)

  # Not In
  @staticmethod
  def notin_(target_col: Column, in_cols: List[Union[int, float, str]]):
    """
    컬럼의 값이 입력된 리스트 중 없는지 확인합니다. ~isin, not in

    :param target_col: 확인하고 싶은 컬럼
    :param in_cols: 값 리스트
    """
    in_cols = [Column.cast(col) for col in in_cols]
    return FunctionalColumn("notin", target_col, *in_cols)

  # Cast
  @staticmethod
  def cast(col: Column, target_type: str) -> FunctionalColumn:
    """
    Column 객체를 지정한 데이터 유형으로 변환합니다.

    :param col: 변환할 대상 Column 객체
    :param target_type: 변환할 데이터 유형
    """
    if target_type not in [
        "bigint", "boolean", "date", "datetime", "double precision", "float",
        "int", "smallint", "real", "char", "varchar", "text", "time"
    ]:
      raise ValueError(f"target_type '{target_type}' is not available in Cast function.")
    return FunctionalColumn("cast", col, target_type=target_type)

  # !!! IMPORTANT NOTE      QueryFunction Marker
  # If add function here, need to implement it's functionality.
  # Search for "QueryFunction Marker" comments


func = QueryFunction
