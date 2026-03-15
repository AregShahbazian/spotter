# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Spotter."""

import os

block_cipher = None

a = Analysis(
    ['webcam.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('model/face_detection_yunet_2023mar.onnx', 'model'),
        ('model/face_recognition_sface_2021dec.onnx', 'model'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'unittest', 'email', 'html', 'http', 'xml',
        'pydoc', 'doctest', 'argparse', 'pdb', 'profile',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spotter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='spotter',
)
