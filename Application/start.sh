#!/usr/bin/env bash
# start.sh — Full setup + production startup

set -e

echo "════════════════════════════════════════"
echo "   🚀 Telegram RAG Bot — Full Startup"
echo "════════════════════════════════════════"

# ── Config ────────────────────────────────────────────────
RAG_PORT=8002
LLM_PORT=8001
BOT_PORT=8000

PID_FILE=".pids"

mkdir -p logs

# ─────────────────────────────────────────────────────────
# 🧠 STEP 0: Ensure uv is installed
# ─────────────────────────────────────────────────────────
echo ""
echo "🔍 Checking for uv..."

if ! command -v uv &> /dev/null
then
    echo "📦 uv not found. Installing..."
    curl -Ls https://astral.sh/uv/install.sh | bash

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv already installed"
fi

# ─────────────────────────────────────────────────────────
# 🧠 STEP 1: Setup Environment (uv)
# ─────────────────────────────────────────────────────────
echo ""
echo "🔧 Setting up Python 3.11 environment..."

if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  uv venv -p 3.11 .venv
else
  echo "✅ Virtual environment already exists"
fi

echo "⚡ Activating environment..."
source .venv/bin/activate

echo "⬆️ Upgrading pip..."
uv pip install --upgrade pip

# (optional) install requirements
if [ -f "requirements.txt" ]; then
  echo "📚 Installing dependencies..."
  uv pip install -r requirements.txt
fi

echo ""
echo "✅ Environment setup complete"

# ─────────────────────────────────────────────────────────
# 🔥 STEP 2: Cleanup
# ─────────────────────────────────────────────────────────
echo ""
echo "🧹 Cleaning old processes..."

kill_port () {
  PORT=$1
  PIDS=$(lsof -ti :$PORT || true)

  if [ -n "$PIDS" ]; then
    echo "🔥 Killing processes on port $PORT..."
    kill -9 $PIDS || true
  else
    echo "✅ Port $PORT is free"
  fi
}

pkill -f rag_server.py || true
pkill -f llm_server.py || true
pkill -f main.py       || true

kill_port $RAG_PORT
kill_port $LLM_PORT
kill_port $BOT_PORT

sleep 2

# ─────────────────────────────────────────────────────────
# ⏳ STEP 3: Wait for Service
# ─────────────────────────────────────────────────────────
wait_for_service () {
  NAME=$1
  URL=$2

  echo "⏳ Waiting for $NAME..."

  for i in {1..120}; do
    RESPONSE=$(curl -s $URL || true)

    if echo "$RESPONSE" | grep -q '"status": *"ok"'; then
      echo "   ✅ $NAME ready"
      return 0
    fi

    sleep 2
  done

  echo "   ❌ $NAME failed to start (check logs/)"
  exit 1
}

# ─────────────────────────────────────────────────────────
# 📚 STEP 4: Start Services
# ─────────────────────────────────────────────────────────

echo ""
echo "📚 Starting RAG server..."
nohup python rag_server.py > logs/rag.log 2>&1 &
RAG_PID=$!
echo "   PID: $RAG_PID"

wait_for_service "RAG" "http://localhost:${RAG_PORT}/health"

echo ""
echo "🤖 Starting LLM server..."
nohup python llm_server.py > logs/llm.log 2>&1 &
LLM_PID=$!
echo "   PID: $LLM_PID"

wait_for_service "LLM" "http://localhost:${LLM_PORT}/health"

echo ""
echo "🚀 Starting Bot server..."
nohup python main.py > logs/bot.log 2>&1 &
BOT_PID=$!
echo "   PID: $BOT_PID"

wait_for_service "Bot" "http://localhost:${BOT_PORT}/health"

# ─────────────────────────────────────────────────────────
# 💾 STEP 5: Save PIDs
# ─────────────────────────────────────────────────────────
echo "$RAG_PID $LLM_PID $BOT_PID" > $PID_FILE

# ─────────────────────────────────────────────────────────
# 🔗 STEP 6: Register Telegram Webhook
# ─────────────────────────────────────────────────────────
echo ""
echo "🔗 Registering Telegram webhook..."

# Guard: token must be set as a Lightning AI Secret (env var)
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
  echo "   ⚠️  TELEGRAM_BOT_TOKEN is not set — skipping webhook registration."
  echo "      Add it via: Lightning AI → Profile → Secrets → + New Secret"
else
  # Guard: studio URL must be set — derive from env or let user override
  # Lightning AI injects the public hostname as LIGHTNING_CLOUDSPACE_HOST
  # e.g.  01kp1bx5vexdgg1d8c4fb0yqad.cloudspaces.litng.ai
  if [ -z "$WEBHOOK_HOST" ]; then
    if [ -n "$LIGHTNING_CLOUDSPACE_HOST" ]; then
      WEBHOOK_HOST="${BOT_PORT}-${LIGHTNING_CLOUDSPACE_HOST}"
    fi
  fi

  if [ -z "$WEBHOOK_HOST" ]; then
    echo "   ⚠️  Cannot determine webhook URL."
    echo "      Set it manually:"
    echo "      export WEBHOOK_HOST=8000-<your-studio-id>.cloudspaces.litng.ai"
    echo "      Then re-run: bash start.sh"
  else
    WEBHOOK_URL="https://${WEBHOOK_HOST}/webhook"
    echo "   URL: ${WEBHOOK_URL}"

    RESPONSE=$(curl -s -X POST \
      "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
      -d "url=${WEBHOOK_URL}")

    if echo "$RESPONSE" | grep -q '"ok":true'; then
      echo "   ✅ Webhook registered successfully"
    else
      echo "   ❌ Webhook registration failed:"
      echo "      $RESPONSE"
    fi

    # Verify
    INFO=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo")
    echo "   ℹ️  Webhook info: $INFO"
  fi
fi

# ─────────────────────────────────────────────────────────
# 🎉 DONE
# ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "   ✅ All services started successfully!"
echo "════════════════════════════════════════"
echo ""
echo "  RAG : http://localhost:${RAG_PORT}/health"
echo "  LLM : http://localhost:${LLM_PORT}/health"
echo "  Bot : http://localhost:${BOT_PORT}/health"
echo ""
echo "  📄 Logs:"
echo "     tail -f logs/rag.log"
echo "     tail -f logs/llm.log"
echo "     tail -f logs/bot.log"
echo "════════════════════════════════════════"