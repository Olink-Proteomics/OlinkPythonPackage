import matplotlib

matplotlib.use("Agg")

import pytest

from olinkanalyze.data import npx_data1, npx_data2


@pytest.fixture(scope="session")
def data1():
    return npx_data1.copy()


@pytest.fixture(scope="session")
def data2():
    return npx_data2.copy()
