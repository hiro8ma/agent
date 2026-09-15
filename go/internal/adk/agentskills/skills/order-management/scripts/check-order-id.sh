#!/bin/sh
set -eu

value=${1-}
case "$value" in
  ORD-*[!0-9]*|ORD-) exit 1 ;;
  ORD-[0-9]*) exit 0 ;;
  *) exit 1 ;;
esac
