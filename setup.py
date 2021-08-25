from setuptools import setup, find_namespace_packages

from daqm import __version__ as version


setup(
    name="DAQM",
    version=version,
    url="https://github.com/linewalks/daqm",
    author="Linewalks",
    author_email="web@linewalks.com",
    description="Data Analysis Query Machine",
    packages=find_namespace_packages(include=["daqm", "daqm.*"]),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "sqlalchemy"
    ]
)
