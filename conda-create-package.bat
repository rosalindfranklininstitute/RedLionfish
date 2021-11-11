REM Script file to automatically run the conda building package

rmdir /s/q conda-built-packages
mkdir conda-built-packages

conda-build --output-folder ./conda-built-packages -c conda-forge conda-recipe
