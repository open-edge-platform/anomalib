# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ('../../backend/src/alembic', 'src/alembic'),  # Alembic migration scripts
    ('../../backend/src/alembic.ini', 'src'),  # Alembic configuration
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

# Collect trackio; as for some reason it also needs package.json
tmp_ret = collect_all("trackio")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect gradio and gradio_client (needed by trackio, requires source .py files)
tmp_ret = collect_all("gradio")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all("gradio_client")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect safehttpx (needed by gradio)
tmp_ret = collect_all("safehttpx")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect groovy (needed by gradio)
tmp_ret = collect_all("groovy")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Collect rfc3987_syntax data files (needed by jsonschema)
try:
    datas += collect_data_files("rfc3987_syntax")
except Exception:
    pass


a = Analysis(
    ['../../backend/src/main.py'],
    pathex=['../../backend/src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook-multiprocessing.py', 'hook-matplotlib.py', 'hook-setup.py'],
    excludes=[
        "comet_ml",
        "open_clip",
        "tkinter",
        "torch.utils.benchmark",
        "torch.testing._internal",
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
        "pdbpp",
        "IPython",
        "ipdb",
    ],
    noarchive=False,
    optimize=0,
)
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geti-inspect-backend',
)
