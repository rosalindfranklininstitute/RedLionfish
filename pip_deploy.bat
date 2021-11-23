@echo off
REM Deply package to Pypi

rem updates
python -m pip install --upgrade setuptools wheel --user

rem This will create two files in /dist folder: a .whl and a .tar.zip (.gz) file with the package
python setup.py sdist bdist_wheel

rem updates
pip install --upgrade twine --user

rem Uploads ( to PyPi by default). This will ask for credentials
twine upload dist/*

Echo Upload to PyPi completed