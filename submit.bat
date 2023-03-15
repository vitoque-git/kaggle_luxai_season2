for %%I in (.) do set CurrDirName=%%~nxI
echo %CurrDirName%
#tar -czf s.%CurrDirName%.tar.gz *
"c:\Program Files\7-Zip\7z.exe" a -ttar -so -an * | "c:\Program Files\7-Zip\7z.exe" a -si s.%CurrDirName%.tar.gz *