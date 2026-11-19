#!/bin/sh
set -eu

VERSION=${1:-0.1.0}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP="$ROOT/build/MLX Stack.app"
BINARY="$APP/Contents/MacOS/MLXStack"
PLIST="$APP/Contents/Info.plist"
ZIP="$ROOT/releases/MLXStack-v$VERSION.zip"

mkdir -p "$APP/Contents/MacOS" "$ROOT/releases"
swiftc -O -parse-as-library "$ROOT/Sources/MLXStack.swift" \
  -o "$BINARY" -framework SwiftUI -framework AppKit
cp "$ROOT/Resources/Info.plist" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$PLIST"
codesign --force --deep --sign - "$APP"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
printf '%s\n' "$ZIP"
