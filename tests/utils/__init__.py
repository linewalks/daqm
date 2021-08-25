

TEST_ORDER_DATA = 100


class WithDoNothing:
  def __enter__(self):
    pass

  def __exit__(self, type, value, traceback):
    pass
