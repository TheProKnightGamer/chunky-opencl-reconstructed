# Shared settings for the harness scripts. BENCH_DIR holds everything recreatable
# (benchmark Chunky home + scenes, binary cache, kernel variants, python venv).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO/benchmark/harness"
BENCH_DIR="${BENCH_DIR:-$HOME/.cache/chunkycl-bench}"
# Chunky home the harness runs with; point at another one to pick a different clDevice.
BENCH_HOME="${BENCH_HOME:-$BENCH_DIR/chunkyhome}"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
CHUNKY_LIB="${CHUNKY_LIB:-$HOME/.chunky/lib}"
# Chunky core + its runtime deps, then the plugin jar (which bundles JOCL). The plugin
# jar is a SNAPSHOT copied by setup.sh: rebuilding the repo while a benchmark runs
# must not swap classes out from under it.
CORE_JAR="${CORE_JAR:-$(ls -t "$CHUNKY_LIB"/chunky-core-2.5.0-SNAPSHOT*.jar | head -1)}"
PLUGIN_JAR="$BENCH_DIR/chunky-opencl.jar"
CP="$(ls "$CHUNKY_LIB"/*.jar | grep -v chunky-core | tr '\n' ':')$CORE_JAR:$PLUGIN_JAR:$BENCH_DIR/classes"
