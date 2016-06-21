#!/bin/bash

make parallel
make ARGS="$1 $2 $3 $4 $5" NP="$6" run
