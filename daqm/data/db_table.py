"""
DBTable 모듈

Database 테이블을 이용한 Data Wrapper, Query
"""

import sqlalchemy
import pandas as pd

from typing import Callable, List, Generator

from daqm.data.columns import (
    Column,
    OperatedColumn,
    ConstantColumn,
    SimpleColumn,
    FunctionalColumn
)
from daqm.data.data import Data, create_temp_table_from_df, get_random_table_name
from daqm.data.query import Query


class DBTableQuery:
  """
  Databse 테이블 Query하기 위한 class

  :param conn:
    데이터가 있는 DB Connection
  :param Query query:
    적용할 Query
  :param str target_table_name:
    쿼리 타겟 테이블 이름. None이면 query의 data를 사용합니다.
  """

  @classmethod
  def query(
      cls,
      conn: sqlalchemy.engine.Connection,
      query: Query
  ) -> str:
    """
    쿼리를 SQL문으로 변환

    :param conn:
      데이터가 있는 DB Connection
    :param Query query:
      적용할 Query

    :return: 변환된 SQL
    :rtype: str
    """
    return cls(conn, query).sql

  def __init__(
      self,
      conn: sqlalchemy.engine.Connection,
      query: Query,
      target_table_name: str = None
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.query = query
    self.conn = conn

    if target_table_name is None:
      self.table_name = self.query.data.to_db(self.conn)
    else:
      self.table_name = target_table_name

    self._parse_query()

  def _check_and_add_column(
      self,
      col: Column
  ):
    """
    컬럼 연산을 위해 필요한 값들을 계산해 query_map에 추가해준다.
    """
    for child_col in col.children:
      self._check_and_add_column(child_col)

    if col in self.query_map:
      return

    if isinstance(col, SimpleColumn):
      table_alias = self.data_table_alias_map[col.table]
      self.query_map[col] = f"{table_alias}.\"{col.target_column_name}\""
    elif isinstance(col, ConstantColumn):
      if isinstance(col.value, str):
        query = f"'{col.value}'"
      elif col.value is None:
        query = "NULL"
      else:
        query = str(col.value)
      self.query_map[col] = query
    elif isinstance(col, FunctionalColumn):
      # NOTE QueryFunction Marker
      # If add new function in QueryFunction, must add it's implementation here.
      query = None
      if col.func in ["sum", "min", "max", "count", "avg"]:
        query = f"{col.func}({self.query_map[col.columns[0]]})"
      elif col.func == "unique":
        if col.options["to_string"]:
          delimiter = col.options["string_delimiter"]
          query = f"string_agg(distinct {self.query_map[col.columns[0]]}, '{delimiter}')"
        else:
          query = f"array_agg(distinct {self.query_map[col.columns[0]]})"
      elif col.func == "nunique":
        query = f"count(distinct {self.query_map[col.columns[0]]})"
      elif col.func == "all":
        if col.options["to_string"]:
          delimiter = col.options["string_delimiter"]
          query = f"string_agg({self.query_map[col.columns[0]]}, '{delimiter}')"
        else:
          query = f"array_agg({self.query_map[col.columns[0]]})"
      elif col.func == "abs":
        query = f"abs({self.query_map[col.columns[0]]})"
      elif col.func == "round":
        query = f"round({self.query_map[col.columns[0]]}::numeric, {col.options['decimals']})"
      elif col.func in ("ceil", "trunc", "floor"):
        query = f"{col.func}({self.query_map[col.columns[0]]}::numeric)"
      elif col.func == "rank":
        query = "rank() over ("
        if col.columns[1] is not None:
          query += f" partition by {self.query_map[col.columns[1]]}"
        query += f" order by {self.query_map[col.columns[0]]})"
        query = query
      elif col.func == "date_diff":
        end_col = self.query_map[col.columns[0]]
        start_col = self.query_map[col.columns[1]]
        query = f"extract(day from ({end_col}::timestamp - {start_col}::timestamp))::int"
      elif col.func == "date_year":
        query = f"extract(year from {self.query_map[col.columns[0]]})::int"
      elif col.func == "date":
        year_col = self.query_map[col.columns[0]]
        month_col = self.query_map[col.columns[1]]
        day_col = self.query_map[col.columns[2]]
        query = f"date({year_col}::text || '-' ||   {month_col}::text || '-' ||  {day_col}::text)"
      elif col.func == "date_delta":
        query = f"cast({self.query_map[col.columns[0]]} || ' days' as interval)"
      elif col.func == "case":
        query = "case"
        for idx in range(0, len(col.columns), 2):
          if idx == len(col.columns) - 1:
            # It's else col
            value_col = self.query_map[col.columns[idx]]
            query += f" else {value_col}"
          else:
            condition_col = self.query_map[col.columns[idx]]
            value_col = self.query_map[col.columns[idx + 1]]

            query += f" when {condition_col} then {value_col}"
        query += " end"
      elif col.func == "coalesce":
        param = ",".join(self.query_map[each_col] for each_col in col.columns)
        query = f"coalesce({param})"
      elif col.func == "isnull":
        query = f"{self.query_map[col.columns[0]]} is null"
      elif col.func == "notnull":
        query = f"{self.query_map[col.columns[0]]} is not null"
      elif col.func == "in":
        in_cols = ",".join(self.query_map[in_col] for in_col in col.columns[1:])
        query = f"{self.query_map[col.columns[0]]} in ({in_cols})"
      elif col.func == "notin":
        in_cols = ",".join(self.query_map[in_col] for in_col in col.columns[1:])
        query = f"{self.query_map[col.columns[0]]} not in ({in_cols})"
      elif col.func == "greatest":
        param = ",".join(self.query_map[each_col] for each_col in col.columns)
        query = f"greatest({param})"
      elif col.func == "least":
        param = ",".join(self.query_map[each_col] for each_col in col.columns)
        query = f"least({param})"
      elif col.func == "and":
        query = " and ".join(self.query_map[each_col] for each_col in col.columns)
      elif col.func == "or":
        query = " or ".join(self.query_map[each_col] for each_col in col.columns)
      elif col.func == "cast":
        if col.options["target_type"] == "datetime":
          cast_type = "timestamp"
        else:
          cast_type = col.options["target_type"]
        query = f"cast({self.query_map[col.columns[0]]} AS {cast_type})"
      else:
        raise NotImplementedError(f"Function {col.func} not implemented for DataFrame.")
      self.query_map[col] = query
    elif isinstance(col, OperatedColumn):
      left_col = self.query_map[col.l_column]
      right_col = self.query_map[col.r_column]

      # NOTE ColumnOperator Marker
      # If add new function in QueryFunction, must add it's implementation here.
      query = None
      if col.operator == "add":
        query = f"{left_col} + {right_col}"
      elif col.operator == "sub":
        query = f"{left_col} - {right_col}"
      elif col.operator == "mul":
        query = f"{left_col} * {right_col}"
      elif col.operator == "div":
        query = f"{left_col} / {right_col}::numeric"
      elif col.operator == "floordiv":
        query = f"floor({left_col} / {right_col})"
      elif col.operator == "lt":
        query = f"{left_col} < {right_col}"
      elif col.operator == "le":
        query = f"{left_col} <= {right_col}"
      elif col.operator == "eq":
        query = f"{left_col} = {right_col}"
      elif col.operator == "ne":
        query = f"{left_col} != {right_col}"
      elif col.operator == "gt":
        query = f"{left_col} > {right_col}"
      elif col.operator == "ge":
        query = f"{left_col} >= {right_col}"
      elif col.operator == "like":
        query = f"{left_col} like {right_col}"
      elif col.operator == "ilike":
        query = f"{left_col} ilike {right_col}"
      elif col.operator == "notlike":
        query = f"{left_col} not like {right_col}"
      elif col.operator == "notilike":
        query = f"{left_col} not ilike {right_col}"
      self.query_map[col] = f"({query})"

  def _parse_query(self):
    """
    Query Parsing
    """
    self.query_map = {}

    # Define Table Alias
    self.data_table_alias_map = {}
    self.data_table_alias_map[self.query.data] = "base"
    for idx, join in enumerate(self.query.join_list):
      data = join[0]
      self.data_table_alias_map[data] = f"join_{idx}"

    select_queries = []
    join_queries = []
    for data, left_on, right_on, how, suffixes in self.query.join_list:
      for col in left_on:
        self._check_and_add_column(col)
      for col in right_on:
        self._check_and_add_column(col)

      table_alias = self.data_table_alias_map[data]
      table_name = data.to_db(self.conn)
      join_query = f"{how} join {table_name} {table_alias} on "
      join_query += " and ".join(
          f"{self.query_map[left_col]} = {self.query_map[right_col]}"
          for left_col, right_col in zip(left_on, right_on))

      join_queries.append(join_query)

      if len(self.query.select_list) == 0:
        left_cols = set(self.query.data.c.all())
        right_cols = set(data.c.all())
        for col in left_cols:
          self._check_and_add_column(col)
        for col in right_cols:
          self._check_and_add_column(col)

        left_on_names = set(col.name for col in left_on)
        right_on_names = set(col.name for col in right_on)
        left_col_names = set(col.name for col in left_cols)
        right_col_names = set(col.name for col in right_cols)
        for col in left_cols:
          if col.name in left_col_names and col.name in right_col_names:  # duplicate
            if col.name in left_on_names and col.name in right_on_names:    # join key column
              query = f"{self.query_map[col]}"
            else:
              query = f"{self.query_map[col]} as \"{col.name}{suffixes[0]}\""
            select_queries.append(query)
          else:
            query = f"{self.query_map[col]} as \"{col.name}\""
            select_queries.append(query)
        for col in right_cols:
          if col.name in left_on_names and col.name in right_on_names:    # join key column
            pass                                              # don't add. added in left_cols loop
          elif col.name in left_col_names and col.name in right_col_names:  # duplicate
            query = f"{self.query_map[col]} as \"{col.name}{suffixes[1]}\""
            select_queries.append(query)
          else:
            query = f"{self.query_map[col]} as \"{col.name}\""
            select_queries.append(query)

    where_queries = []
    for col in self.query.where_list:
      self._check_and_add_column(col)
      query = f"{self.query_map[col]}"
      where_queries.append(query)

    for col in self.query.select_list:
      self._check_and_add_column(col)

      query = f"{self.query_map[col]} as \"{col.name}\""
      select_queries.append(query)

    groupby_queries = []
    for col in self.query.groupby_list:
      self._check_and_add_column(col)
      groupby_queries.append(self.query_map[col])

    orderby_queries = []
    for col in self.query.orderby_list:
      self._check_and_add_column(col)
      desc = "desc" if col.is_desc() else "asc"
      orderby_queries.append(f"{self.query_map[col]} {desc}")

    join_queries = " ".join(join_queries)
    if select_queries:
      select_queries = ", ".join(select_queries)
    else:
      select_queries = "*"
    query = f"""
      select {select_queries}
      from {self.table_name} {self.data_table_alias_map[self.query.data]}
      {join_queries}
    """
    if where_queries:
      query += " where " + " and ".join(where_queries)
    if groupby_queries:
      query += " group by " + ", ".join(groupby_queries)
    if orderby_queries:
      query += " order by " + ", ".join(orderby_queries)

    self.sql = query


def execute_query(conn, query: str):
  """
  conn의 타입에 따라 다른 방식으로 쿼리를 실행합니다.
  """
  if isinstance(conn, sqlalchemy.engine.Connection):
    conn.execute(query)
  else:
    conn.cursor().execute(query)


class DBTableData(Data):
  """
  Database Table Wrapper

  :param conn:
    DB Connection
  :param str table_name:
    테이블 이름
  :param bool is_temp:
    임시 테이블 여부
  """
  @classmethod
  def from_df(
      cls,
      conn: sqlalchemy.engine.Connection,
      df: pd.DataFrame
  ) -> "DBTableData":
    """
    pandas DataFrame을 DB 임시 테이블로 만들고 DBTableData 생성합니다.

    :param conn:
      DB Connection
    :param pandas.DataFrame df:
      pandas DataFrame Object

    :return: 생성된 DBTableData
    :rtype: DBTableData
    """
    table_name = create_temp_table_from_df(conn, df)
    return cls(conn, table_name, is_temp=True)

  @classmethod
  def from_sql(
      cls,
      conn: sqlalchemy.engine.Connection,
      sql: str
  ) -> "DBTableData":
    """
    DB 쿼리 결과를 DBTabledata로 만듭니다.

    :param conn:
      DB Connection
    :param str sql:

    :return: 쿼리 결과를 담은 DBTableData
    :rtype: DBTableData
    """
    table_name = get_random_table_name()
    query = f"create temp table {table_name} as ({sql})"
    execute_query(conn, query)
    return cls(conn, table_name, is_temp=True)

  def __init__(
      self,
      conn: sqlalchemy.engine.Connection,
      table_name: str,
      is_temp: bool = True
  ):
    """
    Initialize self. See help(type(self)) for accurate signature.
    """
    self.conn = conn
    self.table_name = table_name
    self.is_temp = is_temp

    query = f"select * from {self.table_name} limit 0"
    df = pd.read_sql(query, self.conn)
    columns = df.columns
    super().__init__(columns)

  def apply_query(self, query: Query) -> "DBTableData":
    """
    Query를 적용합니다.

    :param Query query:
      적용할 Query Object

    :return: Query가 적용된 Data
    :rtype: DBTableData
    """
    return DBTableData.from_sql(self.conn, DBTableQuery.query(self.conn, query))

  def to_df(self) -> pd.DataFrame:
    """
    Data를 pandas DataFrame으로 변환합니다.

    :return: pandas.DataFrame 형태로 변환된 데이터.
    :rtype: pandas.DataFrame
    """
    return pd.read_sql(f"select * from {self.table_name}", self.conn)

  def to_db(self, conn: sqlalchemy.engine.Connection) -> str:
    """
    Data를 DB의 임시 테이블로 변환합니다.

    :param conn:
      테이블이 생성될 DB Connection

    :return: 생성된 테이블 이름.
    :rtype: str
    """
    if self.conn != conn:
      return super().to_db(conn)
    else:
      return self.table_name

  def rename_column(self, rename_dict: dict) -> "DBTableData":
    """
    컬럼의 이름을 재정의합니다.

    :param dict rename_dict:
      기존 컬럼 이름을 key, 변경할 컬럼 이름을 value로 하는 dict

    :return: 컬럼 이름이 변경된 Data
    :rtype: DBTableData
    """
    new_cols = []
    for col_name in self.columns:
      new_name = rename_dict.get(col_name, col_name)
      new_cols.append(self.c[col_name].label(new_name))
    data = self.query.select(
        new_cols
    ).apply()

    return data

  def copy(self) -> "DBTableData":
    """
    현재 Data를 복사합니다.

    :return: 복사된 Data
    :rtype: DBTableData
    """
    query = f"select * from {self.table_name}"
    return DBTableData.from_sql(self.conn, query)

  def apply_function(
      self,
      func: Callable[[List], List],
      columns: List[str] = None
  ) -> "DBTableData":
    """
    데이터에 함수를 적용합니다.

    :param Callable func:
      데이터 한 줄을 list로 입력 받아 변환된 줄을 list로 리턴하는 함수.
    :param list columns:
      적용 후 컬럼 이름.

    :return: 함수가 적용된 Data
    :rtype: DBTableData
    """
    chunk_size = 4096

    chunk_data_list = []
    table_name = get_random_table_name()

    def make_chunk(is_first):
      """
      chunk_data_list를 DB 테이블에 추가합니다.
      """
      df = pd.DataFrame(chunk_data_list, columns=columns)

      # Pandas  버그  (1.0.3)     NOTE TODO
      # schema가 설정되어 있으면 if_exists가 제대로 작동하지 않는다.
      # 처음에만 schema를 설정하고 그 후는 schema 없이 호출한다.
      if is_first:
        df.to_sql(
            table_name,
            self.conn,
            schema="pg_temp",
            if_exists="fail",
            index=False
        )
      else:
        df.to_sql(
            table_name,
            self.conn,
            if_exists="append",
            index=False
        )

    chunk_count = 0
    for row in self.iter():
      new_row = func(row)
      chunk_data_list.append(new_row)

      if len(chunk_data_list) >= chunk_size:
        chunk_count += 1
        make_chunk(chunk_count == 1)
        chunk_data_list.clear()

    if len(chunk_data_list) > 0:
      chunk_count += 1
      make_chunk(chunk_count == 1)

    return DBTableData(self.conn, table_name)

  def _iter_query(self, query: str) -> Generator[list, None, None]:
    """
    쿼리 결과를 한 줄씩 탐색합니다.

    :return: 한 줄씩 list로 리턴하는 Generator.
    :rtype: generator
    """
    if isinstance(self.conn, sqlalchemy.engine.Connection):
      result_cursor = self.conn.execute(query)
    else:
      result_cursor = self.conn.cursor()
      result_cursor.execute(query)
    while True:
      row = result_cursor.fetchone()
      if row is None:
        break

      yield row
    result_cursor.close()

  def iter(self) -> Generator[list, None, None]:
    """
    데이터를 한 줄씩 탐색합니다.

    :return: 한 줄씩 list로 리턴하는 Generator.
    :rtype: generator
    """
    query = f"select * from {self.table_name}"
    for row in self._iter_query(query):
      yield row

  def count(self) -> int:
    """
    데이터의 row 수를 가져옵니다.

    :return: 데이터 총 row 수
    :rtype: int
    """
    query = f"select count(*) from {self.table_name}"

    count_value = next(self._iter_query(query))[0]
    return count_value
