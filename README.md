# MLX Stack

Native macOS tray and desktop monitor with the complete local-llm backend bundled inside the app. It shows live MLX/llama.cpp metrics, model context, CPU/RAM, processes and logs; it can also load downloaded models from the status-bar menu.

## Install with Homebrew

```console
brew tap mukhanov/local-llm
brew trust mukhanov/local-llm
brew install --cask mlx-stack
open -a "MLX Stack"
```

The cask installs only `MLX Stack.app` into `/Applications`. The backend and `uv` are bundled; no separate local-llm formula, Python installation or terminal setup is required.

On first launch the app automatically creates `~/.ollmlx/mlx-stack-venv` and installs its Python dependencies. This initial setup requires internet access; later launches reuse it. Downloaded models, Hugging Face tokens, client configuration and logs remain in their existing locations.

## Features

- Native SwiftUI desktop panel with one-second updates.
- Status-bar menu for opening the panel, selecting downloaded models and exiting.
- Main/helper tok/s graphs, context history, TTFT, uncached tokens and active generations.
- CPU/core history, memory/swap, process RSS and scrollable logs.
- Click-to-copy client commands.
- Models launched by MLX Stack remain foreground children and stop on Exit.
- Independently launched terminal stacks are observed, never taken over.

## Build from source

Requirements: macOS 14+, Apple Silicon, Swift Command Line Tools and [uv](https://docs.astral.sh/uv/) for the build only. The build bundles `uv` into the app.

```console
./build.sh 0.1.3
open "build/MLX Stack.app"
```

The script also creates `releases/MLXStack-v0.1.3.zip`, the artifact used by the Homebrew cask.

## Runtime

`Contents/Resources/backend` contains the complete Python backend. At launch the app uses its bundled `uv` to prepare `~/.ollmlx/mlx-stack-venv`, install the backend as a non-editable package, and start `python -m local_llm.widget`. Model processes use the same isolated interpreter.

The API listens only on `127.0.0.1:19798` and stops when the app exits. Bootstrap and API diagnostics are written to `~/.ollmlx/logs/mlx-stack.log`.

## License

MIT
