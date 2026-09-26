#!/bin/bash
# Setup: benchmark Chunky home (isolated from ~/.chunky settings), scene links, a
# snapshot of the plugin jar, and the compiled harness. Rerun it after rebuilding the
# plugin jar. Scenes: name=<path to a Chunky scene directory>, loaded from their
# .octree2 dump with spp reset to 0. Usage: setup.sh [name=/path/to/sceneDir ...]
set -e
source "$(dirname "$0")/env.sh"
mkdir -p "$BENCH_DIR"/{chunkyhome/scenes,clcache,classes,variants,out}
cat > "$BENCH_DIR/chunkyhome/chunky.json" <<JSON
{ "minecraftDir": "$HOME/.minecraft", "octreeImplementation": "PACKED",
  "biomeStructureImplementation": "WORLD_TEXTURE_2D", "clDevice": 0 }
JSON
add_scene() {
    local name=$1 src=$2 dst="$BENCH_DIR/chunkyhome/scenes/$1"
    mkdir -p "$dst"
    ln -sf "$(ls "$src"/*.octree2 | head -1)" "$dst/$name.octree2"
    python3 - "$(ls "$src"/*.json | grep -v backup | head -1)" "$dst/$name.json" "$name" <<'PY'
import json, sys
j = json.load(open(sys.argv[1], encoding='utf-8', errors='surrogateescape'))
j['spp'] = 0; j['renderTime'] = 0; j['name'] = sys.argv[3]
json.dump(j, open(sys.argv[2], 'w', encoding='utf-8', errors='surrogateescape'))
PY
    echo "scene $name <- $src"
}
add_scene cltest "$REPO/benchmark/OpenCL_test"
for kv in "$@"; do add_scene "${kv%%=*}" "${kv#*=}"; done
cp "$REPO/build/libs/chunky-opencl.jar" "$PLUGIN_JAR"
"$JAVA_HOME/bin/javac" -d "$BENCH_DIR/classes" -cp "$CP" "$HARNESS/ClBench.java"
echo "harness compiled into $BENCH_DIR/classes"
