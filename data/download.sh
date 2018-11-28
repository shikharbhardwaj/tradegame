#!/bin/bash

IFS=$'\n'

set -f

for i in $(cat < "$1"); do
    wget --no-check-certificate $i
done

