import pytest

from daqm.data.columns import Column, OperatedColumn, ConstantColumn, SimpleColumn, FunctionalColumn
from daqm.utils.type_convert import elem_to_list
from tests.utils import TEST_ORDER_DATA


TEST_COLUMN_NAME = "test_name"


@pytest.mark.run(order=TEST_ORDER_DATA)
class TestColumnInitialize:
  @pytest.mark.parametrize("name_and_data", [("simple", None)])
  def test_simple_column(self, name_and_data):
    col = SimpleColumn(name_and_data[0], name_and_data[1])
    assert col.name == name_and_data[0]
    assert col.table == name_and_data[1]

  @pytest.mark.parametrize("value", [5, 0.123, "abc"])
  def test_constant_column(self, value):
    col = ConstantColumn(value, TEST_COLUMN_NAME)
    assert col.value == value
    assert col.name == TEST_COLUMN_NAME

  @pytest.mark.parametrize("function", ["min", "max", "avg", "unique", "first"])
  @pytest.mark.parametrize("column", [
      SimpleColumn("simple", None),
      ConstantColumn(5)
  ])
  def test_function_column(self, function, column):
    col = FunctionalColumn(function, column, name=TEST_COLUMN_NAME)
    assert col.func == function
    assert col.columns == [column]
    assert col.name == TEST_COLUMN_NAME

  @pytest.mark.parametrize("operator", ["add", "sub", "mul", "div", "mod"])
  @pytest.mark.parametrize("l_column", [
      SimpleColumn("simple", None),
      ConstantColumn(5),
      FunctionalColumn("max", SimpleColumn("simple", None))
  ])
  @pytest.mark.parametrize("r_column", [
      SimpleColumn("simple2", None),
      ConstantColumn(0.123),
      FunctionalColumn("min", ConstantColumn(13))
  ])
  def test_operated_column(self, operator, l_column, r_column):
    col = OperatedColumn(l_column, r_column, operator, TEST_COLUMN_NAME)
    assert col.l_column == l_column
    assert col.r_column == r_column
    assert col.operator == operator
    assert col.name == TEST_COLUMN_NAME


@pytest.mark.run(order=TEST_ORDER_DATA)
@pytest.mark.parametrize("l_column", [
    SimpleColumn("simple", None),
    ConstantColumn(0.123),
    FunctionalColumn("min", ConstantColumn(13)),
    OperatedColumn(
        SimpleColumn("simple", None),
        ConstantColumn(0.123),
        "add"),
    7,
    5.3
])
@pytest.mark.parametrize("r_column", [
    SimpleColumn("simple2", None),
    ConstantColumn(5),
    FunctionalColumn("min", ConstantColumn(0.13)),
    OperatedColumn(
        SimpleColumn("simple2", None),
        ConstantColumn(3),
        "sub"),
    7,
    5.3
])
class TestColumnOperator:
  def _check_operated_col(self, col, operator, l_column, r_column):
    operator = elem_to_list(operator)
    if isinstance(col, Column):
      assert col.operator in operator
    if isinstance(l_column, Column):
      assert col.l_column == l_column
    if isinstance(r_column, Column):
      assert col.r_column == r_column

  def test_add(self, l_column, r_column):
    col = l_column + r_column
    self._check_operated_col(col, "add", l_column, r_column)

    col = r_column + l_column
    self._check_operated_col(col, "add", r_column, l_column)

  def test_sub(self, l_column, r_column):
    col = l_column - r_column
    self._check_operated_col(col, "sub", l_column, r_column)

    col = r_column - l_column
    self._check_operated_col(col, "sub", r_column, l_column)

  def test_mul(self, l_column, r_column):
    col = l_column * r_column
    self._check_operated_col(col, "mul", l_column, r_column)

    col = r_column * l_column
    self._check_operated_col(col, "mul", r_column, l_column)

  def test_div(self, l_column, r_column):
    col = l_column / r_column
    self._check_operated_col(col, "div", l_column, r_column)

    col = r_column / l_column
    self._check_operated_col(col, "div", r_column, l_column)

  def test_floordiv(self, l_column, r_column):
    col = l_column // r_column
    self._check_operated_col(col, "floordiv", l_column, r_column)

    col = r_column // l_column
    self._check_operated_col(col, "floordiv", r_column, l_column)

  def test_lt_gt(self, l_column, r_column):
    col = l_column < r_column
    self._check_operated_col(col, ["lt", "gt"], l_column, r_column)

    col = r_column < l_column
    self._check_operated_col(col, ["lt", "gt"], r_column, l_column)

  def test_le_ge(self, l_column, r_column):
    col = l_column <= r_column
    self._check_operated_col(col, ["le", "ge"], l_column, r_column)

    col = r_column <= l_column
    self._check_operated_col(col, ["le", "ge"], r_column, l_column)

  def test_eq(self, l_column, r_column):
    col = l_column == r_column
    self._check_operated_col(col, "eq", l_column, r_column)

    col = r_column == l_column
    self._check_operated_col(col, "eq", r_column, l_column)

  def test_ne(self, l_column, r_column):
    col = l_column != r_column
    self._check_operated_col(col, "ne", l_column, r_column)

    col = r_column != l_column
    self._check_operated_col(col, "ne", r_column, l_column)
