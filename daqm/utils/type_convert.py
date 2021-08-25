

def elem_to_list(elem, none_to_empty_list=False):
  """
  Change elem to [elem] if elem is not list.
  """
  if none_to_empty_list and elem is None:
    return []
  elif not isinstance(elem, list):
    return [elem]
  else:
    return elem
