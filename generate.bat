@echo off
git submodule update --init --recursive
git checkout HEAD -- deps/premake/
tools\premake5 %* vs2022