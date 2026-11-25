#!/bin/sh
set -eu

VERSION=${1:-0.1.5}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP="$ROOT/build/MLX Stack.app"
BINARY="$APP/Contents/MacOS/MLXStack"
PLIST="$APP/Contents/Info.plist"
ZIP="$ROOT/releases/MLXStack-v$VERSION.zip"
UV=${UV:-$(command -v uv 2>/dev/null || true)}
[ -n "$UV" ] || {
  echo "uv is required to build the standalone app" >&2
  exit 1
}

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/bin" "$ROOT/releases"
swiftc -O -parse-as-library "$ROOT/Sources/MLXStack.swift" \
  "$ROOT/Sources/ModelManager.swift" \
  "$ROOT/Sources/ToolsView.swift" \
  "$ROOT/Sources/MemoryBudget.swift" \
  -o "$BINARY" -framework SwiftUI -framework AppKit
cp "$ROOT/Resources/Info.plist" "$PLIST"
cp "$ROOT/Resources/MLX Stack.icns" "$APP/Contents/Resources/MLX Stack.icns"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/Backend/" "$APP/Contents/Resources/backend/"
cp "$UV" "$APP/Contents/Resources/bin/uv"
cp "$ROOT/Resources/uv-LICENSE-MIT" "$APP/Contents/Resources/bin/uv-LICENSE-MIT"
chmod 755 "$APP/Contents/Resources/bin/uv"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$PLIST"
codesign --force --deep --sign - "$APP"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
printf '%s\n' "$ZIP"
