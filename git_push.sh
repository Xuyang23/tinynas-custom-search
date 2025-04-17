#!/bin/bash

cd /mnt/workspace/lightweight-neural-architecture-search

git add .
git commit -m "🆕 自动提交 $(date +'%Y-%m-%d %H:%M:%S')" || echo "⚠️ 无改动可提交"
git push origin main
