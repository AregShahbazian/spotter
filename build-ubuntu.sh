#!/usr/bin/env bash
#
# Build Spotter .deb package for Ubuntu.
# Run from the project root: ./build-ubuntu.sh
#
set -e

APP_NAME="spotter"
VERSION="1.0.0"
ARCH="amd64"
DEB_NAME="${APP_NAME}_${VERSION}_${ARCH}"
VENV_DIR=".venv"
DIST_DIR="dist"

echo "=== Spotter — Ubuntu package builder ==="
echo ""

# --- 1. Ensure venv + deps ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/5] Virtual environment exists."
fi

echo "[1/5] Installing build dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet opencv-python Pillow pyinstaller

# --- 2. Ensure models are downloaded ---
echo "[2/5] Checking models..."
MODEL_DIR="model"
mkdir -p "$MODEL_DIR"

YUNET="$MODEL_DIR/face_detection_yunet_2023mar.onnx"
SFACE="$MODEL_DIR/face_recognition_sface_2021dec.onnx"

if [ ! -f "$YUNET" ]; then
    echo "  Downloading YuNet model..."
    curl -sL -o "$YUNET" \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fi
if [ ! -f "$SFACE" ]; then
    echo "  Downloading SFace model..."
    curl -sL -o "$SFACE" \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
fi
echo "  Models ready."

# --- 3. Run PyInstaller ---
echo "[3/5] Running PyInstaller..."
"$VENV_DIR/bin/pyinstaller" --noconfirm --clean spotter.spec 2>&1 | tail -5

if [ ! -f "$DIST_DIR/spotter/spotter" ]; then
    echo "ERROR: PyInstaller build failed — dist/spotter/spotter not found."
    exit 1
fi
echo "  PyInstaller build complete."

# --- 4. Convert SVG icon to PNG ---
echo "[4/5] Preparing icon..."
ICON_SVG="assets/spotter.svg"
ICON_PNG="assets/spotter.png"
if [ ! -f "$ICON_PNG" ]; then
    if command -v rsvg-convert &> /dev/null; then
        rsvg-convert -w 256 -h 256 "$ICON_SVG" > "$ICON_PNG"
    elif command -v convert &> /dev/null; then
        convert -background none -resize 256x256 "$ICON_SVG" "$ICON_PNG"
    else
        echo "  WARNING: No SVG converter found (install librsvg2-bin or imagemagick)."
        echo "  Using SVG icon directly — package will work but icon may not display in all launchers."
        ICON_PNG="$ICON_SVG"
    fi
fi
echo "  Icon ready."

# --- 5. Build .deb package ---
echo "[5/5] Building .deb package..."

DEB_ROOT="$DIST_DIR/$DEB_NAME"
rm -rf "$DEB_ROOT"

# Application files
mkdir -p "$DEB_ROOT/opt/spotter"
cp -r "$DIST_DIR/spotter/." "$DEB_ROOT/opt/spotter/"

# Desktop entry
mkdir -p "$DEB_ROOT/usr/share/applications"
cp spotter.desktop "$DEB_ROOT/usr/share/applications/spotter.desktop"

# Icon
mkdir -p "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps"
if [[ "$ICON_PNG" == *.png ]]; then
    cp "$ICON_PNG" "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/spotter.png"
else
    mkdir -p "$DEB_ROOT/usr/share/icons/hicolor/scalable/apps"
    cp "$ICON_SVG" "$DEB_ROOT/usr/share/icons/hicolor/scalable/apps/spotter.svg"
fi

# Symlink in PATH
mkdir -p "$DEB_ROOT/usr/local/bin"
ln -sf /opt/spotter/spotter "$DEB_ROOT/usr/local/bin/spotter"

# Debian control file
mkdir -p "$DEB_ROOT/DEBIAN"
cat > "$DEB_ROOT/DEBIAN/control" << CTRL
Package: spotter
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: libgl1, libglib2.0-0, libsm6, libxext6, libxrender1, libxcb1
Maintainer: Spotter Dev <spotter@localhost>
Description: Live face detection, recognition, and tracking
 Spotter detects and recognizes faces from your webcam in real-time.
 It auto-assigns names, tracks sightings, and shows a live dashboard
 with thumbnails, history, and toast notifications. Supports EN/RU.
Installed-Size: $(du -sk "$DEB_ROOT/opt/spotter" | cut -f1)
CTRL

# Post-install: update icon cache
cat > "$DEB_ROOT/DEBIAN/postinst" << 'POST'
#!/bin/bash
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f /usr/share/icons/hicolor/ 2>/dev/null || true
fi
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications/ 2>/dev/null || true
fi
POST
chmod 755 "$DEB_ROOT/DEBIAN/postinst"

# Post-remove: same cleanup
cat > "$DEB_ROOT/DEBIAN/postrm" << 'POST'
#!/bin/bash
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f /usr/share/icons/hicolor/ 2>/dev/null || true
fi
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications/ 2>/dev/null || true
fi
POST
chmod 755 "$DEB_ROOT/DEBIAN/postrm"

# Build the .deb
dpkg-deb --build --root-owner-group "$DEB_ROOT"

DEB_FILE="$DIST_DIR/${DEB_NAME}.deb"
echo ""
echo "=== BUILD COMPLETE ==="
echo "Package: $DEB_FILE"
echo "Size:    $(du -h "$DEB_FILE" | cut -f1)"
echo ""
echo "To install:"
echo "  sudo dpkg -i $DEB_FILE"
echo ""
echo "To run:"
echo "  spotter              (from terminal)"
echo "  Search 'Spotter'     (in Activities / app launcher)"
echo ""
echo "To uninstall:"
echo "  sudo dpkg -r spotter"
