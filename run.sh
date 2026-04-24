#!/usr/bin/env bash
set -e

BACKEND=${BACKEND:-cpu}

case "$BACKEND" in
  cpu)
    SERVICE=app-cpu
    ;;
  rocm)
    SERVICE=app-rocm
    ;;
  cuda)
    SERVICE=app-cuda
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND (use cpu|rocm|cuda)"
    exit 1
    ;;
esac

echo "Starting service: $SERVICE"
docker compose up --build "$SERVICE"