@setlocal enabledelayedexpansion

set /A b0=0
@set /A b0_base=14460 %0.014460
set /A b0_base=11000 %0.012000

for /L %%i in (0, 1, 240) do (
	set /A b0 = b0_base + %%i * 25
	echo !b0!
	..\x64\Release\CycleBranch2.exe !b0! 13990 12855 13239 13195 12945 13844 16970 > log_25_%%i.txt
)
@pause

