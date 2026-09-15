#!/bin/sh
set -eu

test "$#" -eq 4

for value in "$@"; do
  case "$value" in
    ''|*[!0-9]*) exit 2 ;;
  esac
done

transport=$1
food=$2
activity=$3
total=$4
test "$((transport + food + activity))" -eq "$total"
