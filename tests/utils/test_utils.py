import pytest

from daqm.utils.type_convert import elem_to_list


class TestUtilFunctions:

  @pytest.mark.parametrize("elem", [
      [1],
      [2, 3],
      ["abc"],
      [0.51, 0.1],
      [None, None]
  ])
  def test_elem_to_list_list(self, elem):
    assert elem == elem_to_list(elem)

  @pytest.mark.parametrize("elem", [
      1,
      "abc",
      0.5,
      None
  ])
  def test_elem_to_list_elem(self, elem):
    converted_elem = elem_to_list(elem)
    assert len(converted_elem) == 1
    assert converted_elem[0] == elem

  def test_elem_to_list_empty(self):
    assert len(elem_to_list(None, none_to_empty_list=True)) == 0
