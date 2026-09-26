#!/bin/bash
# Time and compare kernel variants on one scene. Usage: run.sh <scene> variants=... [key=value ...]
source "$(dirname "$0")/env.sh"
scene=$1; shift
[ "$REPO/build/libs/chunky-opencl.jar" -nt "$PLUGIN_JAR" ] && \
    echo "note: build/libs/chunky-opencl.jar is newer than the harness snapshot; rerun setup.sh to use it" >&2
exec "$JAVA_HOME/bin/java" -Xmx8g -Dchunky.home="$BENCH_HOME" -cp "$CP" \
    ClBench sceneDir="$BENCH_HOME/scenes" scene="$scene" cache="$BENCH_DIR/clcache" "$@"
