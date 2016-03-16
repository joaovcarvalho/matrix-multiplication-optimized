#!/bin/bash
printf "Compiling\n"
g++ -msse3 -fopenmp main.c

COLOR='\033[0;32m'
NC='\033[0m' # No Color

printf "Testing with ${COLOR}400x400${NC} mult\n"
./a.out 400 400 400 400

printf "Testing with ${COLOR}800x800${NC} mult\n"
./a.out 800 800 800 800

printf "Testing with ${COLOR}1000x1000${NC} mult\n"
./a.out 1000 1000 1000 1000
