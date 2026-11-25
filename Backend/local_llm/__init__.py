"""local-llm — local MLX stack: a model + OpenAI/Anthropic APIs + a desktop monitor.

Modules:
  config   — paths, ports, env-driven settings
  applog   — persistent app log (~/.mlx-stack/logs/app.log) + crash hooks
  ui       — colored output, hardware info, formatting
  cli      — commands and orchestration: fetch → start → serve until a signal
  monitor  — telemetry readers for the desktop app: ports, processes, logs
  servers  — mlx_lm.server and client-proxy lifecycle + the restart watchdog
  mlxwrap  — mlx_lm.server entry wrapper: tok/s lines + context guard
             (spawned as its own process: `python -m local_llm.mlxwrap`)
  proxy    — client proxy (:4000): OpenAI pass-through + Anthropic bridge
  clients  — pi / omp / Claude Code config writer
  hf       — HuggingFace: DoH DNS workaround, catalog API, cache, download
  models   — model knowledge: name parsing, RAM estimates, hardware fit
  llmfit   — llmfit catalog cache + composite score port for the model catalog
  vendor/  — model impls ahead of mlx_lm releases (used only if missing)

Module names are load-bearing: servers are spawned and pkill-matched by
module path — don't rename them casually.
"""

__version__ = "2.1.2"
