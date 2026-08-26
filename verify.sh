#!/bin/bash
set -e

IMAGE="${IMAGE:-ghalaalnasser/aidc-serving:cpu-v1}"

echo "1. Removing local image..."
docker rm -f serving 2>/dev/null || true
docker rmi "$IMAGE" 2>/dev/null || true

echo "2. Pulling fresh image from Docker Hub..."
docker pull "$IMAGE"

echo "3. Running container..."
docker run -d --name serving -p 8000:8000 -v hf-cache:/home/app/.cache/huggingface "$IMAGE"

echo "4. Waiting for /health..."
until curl -s http://localhost:8000/health | grep -q '"status":"ok"'; do
    sleep 2
done

echo "5. Testing completion..."
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}' > /dev/null

echo ""
echo "GREEN CHECK: PASS"