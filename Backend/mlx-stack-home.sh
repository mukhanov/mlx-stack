#!/bin/sh
# Shared by the API and CLI launchers so the home migration happens before
# either creates logs or starts Python.
mlx_stack_home() {
    STACK_HOME="${MLX_STACK_HOME:-${OLLMLX_HOME:-$HOME/.mlx-stack}}"
    if [ "$STACK_HOME" = "$HOME/.mlx-stack" ] &&
       [ -d "$HOME/.ollmlx" ] && [ ! -L "$HOME/.ollmlx" ]; then
        if [ ! -e "$STACK_HOME" ]; then
            mv "$HOME/.ollmlx" "$STACK_HOME"
        else
            for entry in "$HOME/.ollmlx"/* "$HOME/.ollmlx"/.[!.]*; do
                [ -e "$entry" ] || continue
                name=${entry##*/}
                if [ -e "$STACK_HOME/$name" ]; then
                    mv "$entry" "$STACK_HOME/$name.legacy"
                else
                    mv "$entry" "$STACK_HOME/$name"
                fi
            done
            rmdir "$HOME/.ollmlx" 2>/dev/null || true
        fi
    fi
    if [ "$STACK_HOME" = "$HOME/.mlx-stack" ] &&
       [ ! -e "$HOME/.ollmlx" ]; then
        ln -s "$STACK_HOME" "$HOME/.ollmlx"
    fi
    OLLMLX="$STACK_HOME"
    export OLLMLX
}
