# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ('../../backend/src/alembic', 'src/alembic'),  # Alembic migration scripts
    ('../../backend/src/alembic.ini', 'src'),  # Alembic configuration
    ('../../backend/src/core/model_metadata.yaml', 'core'),  # Model metadata
]
binaries = []
hiddenimports = [
    "aiosqlite",
    "rfc3987_syntax",
    "rfc3987_syntax.utils",
    "rfc3987_syntax.syntax_helpers",
]

# Collect anomalib
tmp_ret = collect_all("anomalib")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect lightning_fabric
tmp_ret = collect_all("lightning_fabric")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect torch
tmp_ret = collect_all("torch")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect torchvision
tmp_ret = collect_all("torchvision")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect openvino (required for model export and conversion)
tmp_ret = collect_all("openvino")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Manually add critical OpenVINO components that collect_all sometimes misses
import openvino
from pathlib import Path
import glob
openvino_path = Path(openvino.__file__).parent
libs_dir = openvino_path / 'libs'

if libs_dir.exists():
    # Add IR frontend - critical for loading .xml models
    ir_frontend_libs = list(libs_dir.glob('libopenvino_ir_frontend.*.dylib'))
    if ir_frontend_libs:
        binaries.append((str(ir_frontend_libs[0]), '.'))
        print(f"Adding IR frontend: {ir_frontend_libs[0].name}")
    else:
        print(f"WARNING: IR frontend not found in {libs_dir}")
    
    # Add OpenVINO plugins (AUTO, CPU, hetero, auto_batch) - required for device selection
    plugin_libs = list(libs_dir.glob('libopenvino_*_plugin.so'))
    for plugin in plugin_libs:
        binaries.append((str(plugin), '.'))
        print(f"Adding OpenVINO plugin: {plugin.name}")
else:
    print(f"WARNING: OpenVINO libs directory not found at {libs_dir}")

# Collect onnx (required for model format used in OpenVINO conversion)
tmp_ret = collect_all("onnx")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['../../backend/src/main.py'],
    pathex=['../../backend/src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["hook-setup.py"],
    excludes=[
        "tkinter",
        "torch.utils.benchmark",
        "torch.testing._internal",
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
        "pdbpp",
        "IPython",
        "ipdb",
        "pyinstaller",
    ],
    noarchive=False,
    optimize=0,
)

# Filter out problematic TBB binaries from Analysis (macOS only)
# These have malformed Mach-O headers and cause install_name_tool/codesign failures
# Keep libtbb.12.dylib (core library needed by OpenVINO), but exclude optional bind/malloc libs
import platform
import os
if platform.system() == "Darwin":
    # Exclude only the problematic TBB bind and malloc libraries, keep the core libtbb
    problematic_tbb_libs = ['libtbbbind', 'libtbbmalloc']
    
    def is_problematic_tbb(name):
        basename = os.path.basename(name)
        return any(basename.startswith(lib) for lib in problematic_tbb_libs)
    
    # Filter from binaries
    a.binaries = [b for b in a.binaries if not is_problematic_tbb(b[0])]
    # Filter from datas (symlinks/data files can end up here too)
    a.datas = [d for d in a.datas if not is_problematic_tbb(d[0])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="geti-inspect-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="geti-inspect-backend",
)