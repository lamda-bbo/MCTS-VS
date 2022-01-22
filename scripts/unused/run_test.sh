#!/bin/bash

func_list=(Hopper Walker2d)
for func in ${func_list[@]}
do
    if [ "$func" = "Hopper" ]
    then
        echo "111"
    elif [ "$func" = "Walker2d" ]
    then
        echo "222"
    else
        echo "333"
    fi
done