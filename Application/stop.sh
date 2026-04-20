#!/usr/bin/env bash

# stop.sh — Stop all running services started by start.sh
# Usage: bash stop.sh

set -e

PID_FILE=".pids"

echo "════════════════════════════════════════"
echo "  Telegram RAG Bot — Stopping services"
echo "════════════════════════════════════════"

# ── Config (must match start.sh exactly) ─────────────────
RAG_PORT=8002
LLM_PORT=8001
BOT_PORT=8000

# ── Check PID file ────────────────────────────────────────
if [ ! -f "$PID_FILE" ]; then
  echo "⚠️  No PID file found. Attempting cleanup via ports..."
else
  read RAG_PID LLM_PID BOT_PID < "$PID_FILE"
fi

echo "🛑 Stopping services..."

# ── Function: graceful stop ───────────────────────────────
stop_process () {
  PID=$1
  NAME=$2

  if [ -z "$PID" ]; then
    return
  fi

  if ps -p $PID > /dev/null 2>&1; then
    echo "   Stopping $NAME (PID: $PID)..."
    kill $PID

    for i in {1..5}; do
      if ! ps -p $PID > /dev/null 2>&1; then
        echo "   ✅ $NAME stopped"
        return
      fi
      sleep 1
    done

    echo "   ⚠️  $NAME still running → force killing..."
    kill -9 $PID || true
  else
    echo "   ⚠️  $NAME (PID: $PID) not running"
  fi
}

# ── Stop using PID file (if exists) ───────────────────────
if [ -f "$PID_FILE" ]; then
  stop_process $BOT_PID "Bot server"
  stop_process $LLM_PID "LLM server"
  stop_process $RAG_PID "RAG server"
fi

# ── Kill anything still bound to ports ────────────────────
echo ""
echo "🧹 Cleaning up ports..."

cleanup_port () {
  PORT=$1
  NAME=$2

  PIDS=$(lsof -ti :$PORT || true)

  if [ -n "$PIDS" ]; then
    echo "   🔥 Killing processes on port $PORT ($NAME)..."
    kill -9 $PIDS || true
  else
    echo "   ✅ Port $PORT is free"
  fi
}

cleanup_port $BOT_PORT "Bot"
cleanup_port $LLM_PORT "LLM"
cleanup_port $RAG_PORT "RAG"

# ── Remove PID file ───────────────────────────────────────
rm -f "$PID_FILE"

echo ""
echo "════════════════════════════════════════"
echo "  ✅ All services stopped & ports freed"
echo "════════════════════════════════════════"