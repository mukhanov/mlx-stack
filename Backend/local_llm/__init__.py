"""local-llm — the MLX Stack backend: a local model behind OpenAI and
Anthropic APIs on localhost.

Modules:
  config       — paths, ports, launch settings and their environment variables
  net          — the one loopback address everything binds, the web-page guard
  applog       — persistent app log (~/.mlx-stack/logs/app.log) + crash hooks
  ui           — colored output, hardware info, formatting
  cli          — commands and orchestration: fetch → start → serve until a signal
  servers      — model-server and client-proxy lifecycle + the restart watchdog
  mlxwrap      — mlx_lm.server entry wrapper: tok/s lines + context guard
                 (spawned as its own process: `python -m local_llm.mlxwrap`)
  proxy        — client proxy (:4000): OpenAI pass-through + Anthropic bridge
  widget       — the app's local API (:19798): snapshot, catalog, load/stop
  hf, llamacpp — Hugging Face downloads and cache; the llama-server for GGUF
  model_catalog, llmfit, models — the catalog and its hardware-fit scoring
  clients      — pi / omp / opencode / Claude Code config writer
  monitor      — telemetry readers for the app's Monitor
  vendor/      — model impls ahead of mlx_lm releases (used only if missing)

Module names are load-bearing: servers are spawned and matched by module
path — don't rename them casually.
"""

__version__ = "2.2.0"
