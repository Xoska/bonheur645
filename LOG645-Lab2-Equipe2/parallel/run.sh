#!/bin/bash

make parallel
make ARGS="$1 $2 $3" run
