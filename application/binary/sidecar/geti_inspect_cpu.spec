# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
import platform
import glob
datas = [
    ('../../backend/src/alembic', 'alembic'),  # Alembic migration scripts
    ('../../backend/src/alembic.ini', '.'),  # Alembic configuration
    ('../../backend/src/core/model_metadata.yaml', 'core'),  # Model metadata
]

# Add all binaries
## Linux/MacOS
if platform.system() == "Linux":
    binaries = [(so, '.') for so in glob.glob('../../backend/.venv/lib/**/*.so.*', recursive=True)]
elif platform.system() == "Darwin":
    binaries = [(so, '.') for so in glob.glob('../../backend/.venv/lib/**/*.dylib', recursive=True)]
else:
    binaries = [(dll, '.') for dll in glob.glob('../../backend/.venv/lib/**/*.dll', recursive=True)]

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
