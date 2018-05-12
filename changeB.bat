@setlocal enabledelayedexpansion

set /A b0=0
set /A b0_base=14460

for /L %%i in (1, 1, 10) do (
	set /A b0 += b0_base + %%i * 100
	echo !b0!
	..\x64\Release\CycleBranch2.exe !b0! 13990 12855 13239 13195 12945 13844 16970 > log_%%i.txt
)
@pause
