Rem Get the output for a simple python code

rem test
rem for /f "tokens=*" %%i in ('python --version') do set REIKNAPATH=%%i

rem tokens=* processes line by line

FOR /F "tokens=*" %%i IN ('python -c "import importlib; print(importlib.util.find_spec('reikna').submodule_search_locations[0])" ') do set REIKNAPATH=%%i

echo REIKNAPATH = %REIKNAPATH%

rem Create correct pyinstaller spec file
rem pyi-makespec test_and_benchm.py --add-data "%REIKNAPATH%;reikna"
rem pyi-makespec test_and_benchm.py --add-data "%REIKNAPATH%;reikna" --exclude-module matplotlib

pyi-makespec test_and_benchm.py --onefile --add-data "%REIKNAPATH%;reikna" --exclude-module matplotlib --exclude-module PyQt5

pyinstaller --clean --noconfirm test_and_benchm.spec
