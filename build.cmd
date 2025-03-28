@echo off
setlocal

for /d %%x in (*) do (
    echo %%x
    pushd %%x
    nmake.exe /nologo
    popd
)
