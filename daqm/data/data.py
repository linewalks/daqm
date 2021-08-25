"""
Data 모듈

Data를 다루기 위한 Abstraat class와 공통 함수
"""


import pandas as pd
import warnings
import sqlalchemy
import uuid

from abc import abstractmethod
from typing import List, Callable, Dict, Generator

from daqm.data.columns import ColumnContainer
from daqm.data.query import Query
from daqm.utils.type_convert import elem_to_list


def get_random_table_name() -> str:
  """
  랜덤 테이블 이름을 가져옵니다.
  uuid4를 사용하며 t_{uuid4} 형태로 만들어집니다.

  :return: 랜덤 테이블 이름
  :rtype: str
  """
  return "t_" + str(uuid.uuid4()).replace("-", "")


def create_temp_table_from_df(
    conn,
    df: pd.DataFrame
) -> str:
  """
  pandas DataFrame으로 conn에 랜덤한 이름의 임시 테이블을 생성합니다.

  :return: 생성된 테이블 이름
  :rtype: str
  """
  table_name = get_random_table_name()
  df.to_sql(
      table_name,
      con=conn,
      if_exists="fail",
      schema="pg_temp",
      index=False
  )
  return table_name


class Data:
  """
  Abstract Data class

  데이터를 다루기 위한 추상 클래스

  :param list columns:
    컬럼 이름
  """
  def __init__(self, columns: List[str]):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.columns = columns
    self.c = ColumnContainer(self.columns, self)

  def check_data_columns(
      self,
      required_cols: List[str],
      optional_cols: List[str] = []
  ) -> List[str]:
    """
    데이터 컬럼을 확인합니다.

    :param list required_cols:
      필수 컬럼.
    :param list optional_cols:
      선택 컬럼.

    :return:
      선택 컬럼 중 존재하는 컬럼과 필수 컬럼을 합친 리스트.
    :rtype: list

    :raise ValueError: required_cols 중 없는 컬럼이 있을 때.
    :raise UserWarning: required_cols도, optional_cols도 아닌 컬럼이 있을 때.
    """
    given_cols_set = set(self.columns)
    required_cols_set = set(required_cols)
    optional_cols_set = set(optional_cols)

    total_cols_set = required_cols_set.union(optional_cols_set)

    if not given_cols_set.issuperset(required_cols_set):
      missing_cols_set = required_cols_set.difference(given_cols_set)
      raise ValueError(f"Required columns {list(missing_cols_set)} missing.")

    not_supported_cols_set = given_cols_set.difference(total_cols_set)
    if not_supported_cols_set:
      warnings.warn(f"Not supported columns {list(not_supported_cols_set)} are given. May cause problems.")

    optional_intersection_cols_set = given_cols_set.intersection(optional_cols_set)
    return required_cols + list(optional_intersection_cols_set)

  @property
  def query(self) -> Query:
    """
    쿼리를 시작합니다.

    :return: Query Object
    :rtype: Query

    :example: 다음과 같이 사용합니다.

    .. code-block:: python

      new_data = data.query.select(data.c.col1, data.c.col2).apply()

    """
    return Query(self)

  @abstractmethod
  def apply_query(self, query: Query):
    """
    Query를 적용합니다.

    :param Query query:
      적용할 Query Object
    """
    pass

  @abstractmethod
  def to_df(self) -> pd.DataFrame:
    """
    Data를 pandas DataFrame으로 변환합니다.

    :return: pandas.DataFrame 형태로 변환된 데이터.
    :rtype: pandas.DataFrame
    """
    pass

  def to_db(self, conn: sqlalchemy.engine.Connection) -> str:
    """
    Data를 DB의 임시 테이블로 변환합니다.

    :param conn:
      테이블이 생성될 DB Connection

    :return: 생성된 테이블 이름.
    :rtype: str
    """
    return create_temp_table_from_df(conn, self.to_df())

  @abstractmethod
  def rename_column(self, rename_dict: Dict[str, str]) -> "Data":
    """
    컬럼의 이름을 재정의합니다.

    :param dict rename_dict:
      기존 컬럼 이름을 key, 변경할 컬럼 이름을 value로 하는 dict

    :return: 컬럼 이름이 변경된 Data
    :rtype: Data
    """
    pass

  @abstractmethod
  def copy(self) -> "Data":
    """
    현재 Data를 복사합니다.

    :return: 복사된 Data
    :rtype: Data
    """
    pass

  @abstractmethod
  def apply_function(
      self,
      func: Callable[[List], List],
      columns: List[str] = None
  ) -> "Data":
    """
    데이터에 함수를 적용합니다.

    :param Callable func:
      데이터 한 줄을 list로 입력 받아 변환된 줄을 list로 리턴하는 함수.
    :param list columns:
      적용 후 컬럼 이름.

    :return: 함수가 적용된 Data
    :rtype: Data
    """
    pass

  @abstractmethod
  def iter(self) -> Generator[list, None, None]:
    """
    데이터를 한 줄씩 탐색합니다.

    :return: 한 줄씩 list로 리턴하는 Generator.
    :rtype: generator
    """
    pass

  @abstractmethod
  def count(self) -> int:
    """
    데이터의 row 수를 가져옵니다.

    :return: 데이터 총 row 수
    :rtype: int
    """
    pass
