@echo off
rem Idempotent zip rebuild. The lock dir makes a doubled shell launch a no-op.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set B=scratchpad\row55_l2i8s4e1c_bundle
set Z=scratchpad\row55_l2i8s4e1c_bundle.zip
mkdir scratchpad\_zip50.lock 2>nul
if errorlevel 1 (
  echo SKIP: another zip instance holds the lock
  exit /b 0
)
del /q "%Z%" 2>nul
powershell -NoProfile -Command "Compress-Archive -Path '%B%' -DestinationPath '%Z%' -CompressionLevel Optimal"
echo ZIP RC=%errorlevel%
rmdir scratchpad\_zip50.lock
powershell -NoProfile -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $z=Get-Item '%Z%'; $c=[System.IO.Compression.ZipFile]::OpenRead($z.FullName); 'zip bytes: ' + $z.Length + '  entries: ' + $c.Entries.Count; $c.Dispose()"
echo ZIP DONE
