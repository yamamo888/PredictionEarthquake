@setlocal enabledelayedexpansion

set /A b0=0
set /A b0_base=11000

for /L %%i in (0, 1, 1200) do (
	set /A b0 = b0_base + %%i * 5
	echo !b0!
	..\x64\Release\CycleBranch2.exe !b0! 13990 12855 13239 13195 12945 13844 16970 > log_5_%%i.txt
)
@pause

