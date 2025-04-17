#!/bin/bash

# 切换到项目目录
cd /mnt/workspace/lightweight-neural-architecture-search

# 初始化 Git 仓库（如果还没 init）
git init

# 设置用户信息
git config --global user.name "Xuyang23"
git config --global user.email "s232184@dtu.dk"

# 设置远程地址（自动注入 GitHub Token）
git remote remove origin 2>/dev/null
git remote add origin https://Xuyang23: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@github.com/Xuyang23/tinynas-custom-search.git

echo "✅ Git 初始化完成"
