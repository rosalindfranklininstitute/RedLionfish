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

REM Script file to automatically run the conda building package
REM Note: This will not upload or update the conda-forge package

rmdir /s/q conda-built-packages
mkdir conda-built-packages

conda-build --output-folder ./conda-built-packages -c conda-forge conda-recipe
