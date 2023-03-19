luxai-s2 ./currentCode/main.py  ./_bots/v102b/main.py -o replay.json -v 10 -s 6 | tee output.log
grep -E "TCFAIL|Traceback|Agent likely errored out" output.log
