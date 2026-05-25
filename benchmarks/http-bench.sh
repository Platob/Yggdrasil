#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-3333}"
BASE="http://localhost:$PORT"
DURATION="${BENCH_DURATION:-10}"
CONNECTIONS="${BENCH_CONNECTIONS:-50}"
PIPELINE="${BENCH_PIPELINE:-5}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Yggdrasil HTTP Benchmarks"
echo "  Target: $BASE | Duration: ${DURATION}s | Connections: $CONNECTIONS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_bench() {
  local label="$1"
  local path="$2"
  echo "── $label [$path] ──"
  autocannon -c "$CONNECTIONS" -d "$DURATION" -p "$PIPELINE" "$BASE$path" 2>&1 \
    | grep -E 'Stat|Req/Sec|Bytes/Sec|requests|Latency|Avg|Min|Max|2xx|non' | head -10
  echo ""
}

check_headers() {
  local path="$1"
  echo "  Headers for $path:"
  curl -sI "$BASE$path" | grep -iE 'cache-control|etag|x-cache|x-content|x-frame|x-powered|content-encoding|vary' | sed 's/^/    /'
  echo ""
}

echo "╔═══════════════════════════════════════╗"
echo "║        RESPONSE HEADER AUDIT          ║"
echo "╚═══════════════════════════════════════╝"
echo ""
for path in / /bot /api/config /api/health /api/cache/dashboard; do
  check_headers "$path"
done

echo "╔═══════════════════════════════════════╗"
echo "║          THROUGHPUT BENCHMARKS        ║"
echo "╚═══════════════════════════════════════╝"
echo ""

run_bench "Static: Welcome page" "/"
run_bench "Static: Bot dashboard" "/bot"
run_bench "API: Config (cacheable)" "/api/config"
run_bench "API: Health (dynamic)" "/api/health"
run_bench "API: Dashboard cache" "/api/cache/dashboard"

echo "╔═══════════════════════════════════════╗"
echo "║         RESPONSE SIZE AUDIT           ║"
echo "╚═══════════════════════════════════════╝"
echo ""
for path in / /bot /bot/execute /bot/chat /bot/network /msg /api/config /api/health; do
  raw=$(curl -s "$BASE$path" | wc -c)
  gzip=$(curl -s -H "Accept-Encoding: gzip" "$BASE$path" | wc -c)
  ratio=$( [ "$raw" -gt 0 ] && echo "scale=1; $gzip * 100 / $raw" | bc || echo "0" )
  printf "  %-25s %6d bytes  →  %6d gzip (%s%%)\n" "$path" "$raw" "$gzip" "$ratio"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Benchmark complete"
echo "═══════════════════════════════════════════════════════════════"
