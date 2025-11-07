@echo off
git checkout HEAD -- deps/premake/ 2>nul
tools\premake5 %*