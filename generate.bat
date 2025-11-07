@echo off
if not exist deps\premake\ (
    git submodule update --init --recursive
)
git checkout HEAD -- deps/premake/ 2>nul
tools\premake5 %*