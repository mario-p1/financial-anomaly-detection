FROM mcr.microsoft.com/devcontainers/python:3-3.13-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY default_claude_settings.json /home/vscode/.claude/settings.json
COPY post-create.sh /home/vscode/