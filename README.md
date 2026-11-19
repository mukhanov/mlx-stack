# MLX Stack

Native macOS tray and desktop monitor for [local-llm](https://github.com/mukhanov/local-llm). It shows live MLX/llama.cpp metrics, model context, CPU/RAM, processes and logs; it can also load downloaded models from the status-bar menu.

## Install with Homebrew

```console
brew tap mukhanov/local-llm
brew trust mukhanov/local-llm
brew install --cask mlx-stack
open -a "MLX Stack"
```

The cask installs `MLX Stack.app` into `/Applications` and installs local-llm as a dependency. Initialize local-llm once with `local-llm` before loading models.

## Features

- Native SwiftUI desktop panel with one-second updates.
- Status-bar menu for opening the panel, selecting downloaded models and exiting.
- Main/helper tok/s graphs, context history, TTFT, uncached tokens and active generations.
- CPU/core history, memory/swap, process RSS and scrollable logs.
- Click-to-copy client commands.
- Models launched by MLX Stack remain foreground children and stop on Exit.
- Independently launched terminal stacks are observed, never taken over.

## Build from source

Requirements: macOS 14+, Apple Silicon, Swift Command Line Tools.

```console
./build.sh 0.1.2
open "build/MLX Stack.app"
```

The script also creates `releases/MLXStack-v0.1.2.zip`, the artifact used by the Homebrew cask.

## Runtime

The app starts `~/.ollmlx/venv/bin/python -m local_llm.widget` as its snapshot API child. The API listens only on `127.0.0.1:19798` and stops when the app exits.

## License

MIT
