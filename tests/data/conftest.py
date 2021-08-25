import pytest
import warnings

from sqlalchemy import create_engine


def pytest_addoption(parser):
  parser.addini("data_test_uri", "DataDBTable test connection URI")


@pytest.fixture(scope="session")
def conn(request):
  uri = request.config.getini("data_test_uri")
  if not uri:
    pytest.skip("data test uri is not set.")
    return

  engine = create_engine(uri)
  try:
    conn = engine.connect()
    yield conn
    conn.close()
  except Exception as e:
    warnings.warn(f"Connecting data test uri Failed. DBTable test will be sipped. {e}")
    yield None

  engine.dispose()