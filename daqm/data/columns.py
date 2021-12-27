"""
Column

Data에서 사용할 컬럼을 정의합니다.
"""

import copy
import numpy as np

from typing import Union, Optional, List


class Column:
  """
  Base Column

  :param str name:
    컬럼 이름
  """
  def __init__(
      self,
      name: Optional[str] = None
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.name = name if name else ""

    # Only used when column in orderby clause
    self._desc = False

  def is_constant(self) -> False:
    """
    상수를 나타내는 컬럼인지 여부.
    """
    return False

  @property
  def children(self) -> List["Column"]:
    """
    자식 컬럼. 연산 등을 위해 사용하는 컬럼.
    """
    return []

  def label(self, name: str) -> "Column":
    """
    컬럼에 이름을 지정한 새 컬럼을 리턴합니다.
    :param str name: 이름
    """
    col = copy.copy(self)
    col.name = name
    return col

  def desc(self) -> "Column":
    """
    컬럼을 내림차순 정렬로 설정한 새 컬럼을 리턴합니다.
    """
    col = copy.copy(self)
    col._desc = True
    return col

  def is_desc(self) -> bool:
    """
    내림차순 정렬 여부
    """
    return self._desc

  def __repr__(self):
    """Column repr"""
    return f"{self.__class__.__name__} {self.name}"

  def __hash__(self):
    """Column hash"""
    return id(self)

  @staticmethod
  def cast(other: Union[np.number, str, bool, "Column"]) -> "Column":
    """
    숫자, 문자열, bool 값을 컬럼으로 변환

    :type other: number, str, bool or Column
    :param other: 변환할 컬럼 혹은 값

    :return: 변환된 Column
    :rtype: Column
    """
    if np.issubdtype(type(other), np.number) or isinstance(other, str) or isinstance(other, bool):
      other = ConstantColumn(other)
    return other

  def _create_operator(self, other: "Column", operator: str, expression: str):
    """
    다른 컬럼과 연산되는 OperatedColumn을 생성합니다.
    """
    return OperatedColumn(self,
                          other,
                          operator,
                          f"({self.name} {expression} {other.name})")

  def _operator_pre_check(self, other):
    """
    연산전 타입 체크 공통 함수
    """
    other = self.cast(other)
    if not isinstance(other, Column):
      raise TypeError(f"Not supported type {type(other)}")
    return other

  # !!! IMPORTANT NOTE      ColumnOperator Marker
  # If add operator here, need to implement it's functionality.
  # Search for "ColumnOperator Marker" comments

  # this is for checking Column object
  def is_(self, other):
    """Column is"""
    return self is other

  def __add__(self, other):
    """Column __add__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "add", "+")

  def __radd__(self, other):
    """Column __radd__"""
    other = self._operator_pre_check(other)
    return other.__add__(self)

  def __sub__(self, other):
    """Column __sub__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "sub", "-")

  def __rsub__(self, other):
    """Column __rsub__"""
    other = self._operator_pre_check(other)
    return other.__sub__(self)

  def __mul__(self, other):
    """Column __mul__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "mul", "*")

  def __rmul__(self, other):
    """Column __rmul__"""
    other = self._operator_pre_check(other)
    return other.__mul__(self)

  def __truediv__(self, other):
    """Column __truediv__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "div", "/")

  def __rtruediv__(self, other):
    """Column __rtruediv__"""
    other = self._operator_pre_check(other)
    return other.__truediv__(self)

  def __floordiv__(self, other):
    """Column __floordiv__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "floordiv", "//")

  def __rfloordiv__(self, other):
    """Column __rfloordiv__"""
    other = self._operator_pre_check(other)
    return other.__floordiv__(self)

  def __lt__(self, other):
    """Column __lt__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "lt", "<")

  def __le__(self, other):
    """Column __le__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "le", "<=")

  def __eq__(self, other):
    """Column __eq__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "eq", "==")

  def __ne__(self, other):
    """Column __ne__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "ne", "!=")

  def __gt__(self, other):
    """Column __gt__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "gt", ">")

  def __ge__(self, other):
    """Column __ge__"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "ge", ">=")

  def like(self, other):
    """Column like"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "like", "like")

  def ilike(self, other):
    """Column ilike"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "ilike", "ilike")

  def notlike(self, other):
    """Column not like"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "notlike", "notlike")

  def notilike(self, other):
    """Column not ilike"""
    other = self._operator_pre_check(other)
    return self._create_operator(other, "notilike", "notilike")

  # !!! IMPORTANT NOTE      ColumnOperator Marker
  # If add operator here, need to implement it's functionality.
  # Search for "ColumnOperator Marker" comments


class OperatedColumn(Column):
  """
  연산 컬럼

  :param Column l_column:
    연산자 왼쪽 컬럼
  :param Column r_column:
    연산자 오른쪽 컬럼
  :param str operator:
    연산자명
  :param str name:
    컬럼 이름
  """
  def __init__(
      self,
      l_column: Optional["Column"],
      r_column: Optional["Column"],
      operator: Optional[str],
      name: Optional[str] = None
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    super().__init__(name)
    self.l_column = l_column
    self.r_column = r_column
    self.operator = operator

  @property
  def children(self) -> List[Column]:
    """
    자식 컬럼. 연산자 좌우 컬럼.
    """
    return [self.l_column, self.r_column]


class ConstantColumn(Column):
  """
  상수 컬럼

  :type value: np.number, str, bool, None
  :param value:
    상수 값
  :param str name:
    컬럼 이름
  """
  def __init__(
      self,
      value: Union[np.number, str, bool, None],
      name: str = None
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    if name is None:
      if isinstance(value, str):
        name = f"'{value}'"
      else:
        name = str(value)
    super().__init__(name)
    self.value = value

  def is_constant(self):
    """
    상수를 나타내는 컬럼인지 여부.
    """
    return True


class SimpleColumn(Column):
  """
  단순 컬럼. 데이터의 한 컬럼을 의미합니다.

  :param str name:
    컬럼 이름
  :param Data table:
    컬럼이 있는 데이터
  """
  def __init__(
      self,
      name: str,
      table: Optional["Data"]
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    super().__init__(name)
    self.target_column_name = name
    self.table = table


class FunctionalColumn(Column):
  """
  함수 컬럼

  :param str func:
    함수 이름
  :param Column columns:
    함수에 필요한 컬럼들
  :param bool is_agg:
    Aggregate 함수 여부
  :param str name:
    컬럼 이름
  """
  def __init__(
      self,
      func: str,
      *columns: Column,
      is_agg: bool = False,
      name: Optional[str] = None,
      **kwargs
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    columns = [Column.cast(col) for col in columns]
    if name is None:
      column_names = " ".join(column.name for column in columns)
      name = f"{func}({column_names})"
    super().__init__(name)
    self.func = func
    self.is_agg = is_agg
    self.columns = columns
    self.options = dict(kwargs)

  @property
  def children(self) -> List[Column]:
    """
    자식 컬럼. 함수에 필요한 컬럼들.
    """
    return self.columns


class ColumnContainer:
  """
  컬럼 Container

  :param list columns:
    컬럼 이름 리스트
  :param Data data:
    컬럼이 속한 데이터
  """
  def __init__(self, columns: List[str], data=None):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.columns = []
    for col_name in columns:
      col = SimpleColumn(col_name, data)
      self.columns.append(col)
      setattr(self, col_name, col)

  def __getitem__(self, key):
    """ColumnContainer __getitem__"""
    return getattr(self, key)

  def all(self) -> List[Column]:
    """
    컬럼 전체를 가져옵니다
    """
    return self.columns


def and_(*columns):
  return FunctionalColumn("and", *columns)


def or_(*columns):
  return FunctionalColumn("or", *columns)
