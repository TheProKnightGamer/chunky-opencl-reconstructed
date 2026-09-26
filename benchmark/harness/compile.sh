#!/bin/bash
# Build kernel variants into the binary cache: one core, one variant at a time, no GPU
# work beyond creating a context. Usage: compile.sh variants=... [names=...]
source "$(dirname "$0")/env.sh"
exec nice -n 10 "$JAVA_HOME/bin/java" -Xmx2g -Dchunky.home="$BENCH_HOME" -cp "$CP" \
    ClBench compileOnly=1 cache="$BENCH_DIR/clcache" "$@"
