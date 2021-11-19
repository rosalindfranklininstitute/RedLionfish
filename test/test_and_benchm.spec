# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['test_and_benchm.py'],
             pathex=['C:\\Users\\Luis\\Desktop\\RedLionfish-git\\test'],
             binaries=[],
             datas=[('C:\\Users\\Luis\\miniconda3\\envs\\dev\\lib\\site-packages\\reikna', 'reikna')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['matplotlib', 'PyQt5'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='test_and_benchm',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
