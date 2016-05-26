#!/bin/bash

make sequential
make ARGS="$1 $2 $3" run
