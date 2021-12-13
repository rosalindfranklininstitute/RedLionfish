@REM Copyright 2021 Rosalind Franklin Institute

@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at

@REM     http://www.apache.org/licenses/LICENSE-2.0

@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.

@echo off

REM Deploy package to Pypi

rem updates
python -m pip install --upgrade setuptools wheel --user

rem This will create two files in /dist folder: a .whl and a .tar.zip (.gz) file with the package
python setup.py sdist bdist_wheel

rem updates
pip install --upgrade twine --user

rem Uploads ( to PyPi by default). This will ask for credentials
twine upload dist/*

Echo Upload to PyPi completed