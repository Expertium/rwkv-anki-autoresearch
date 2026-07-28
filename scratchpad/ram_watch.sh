#!/bin/bash
# Warn while there is still time to act: the 2026-07-28 analysis found every unexplained hang
# sat at 56-63 GB on this 64 GB box. Fires once per threshold crossing, then exits at 60 GB.
CSV="C:/Users/Andrew/hang-monitor/flight_$(date +%Y%m%d).csv"
warned52=0; warned56=0
for i in $(seq 1 900); do
  ram=$(tail -1 "$CSV" 2>/dev/null | cut -d, -f3)
  ok=$(echo "$ram" | grep -E '^[0-9.]+$')
  if [ -n "$ok" ]; then
    hi52=$(echo "$ram >= 52" | bc -l 2>/dev/null)
    hi56=$(echo "$ram >= 56" | bc -l 2>/dev/null)
    hi60=$(echo "$ram >= 60" | bc -l 2>/dev/null)
    if [ "$hi60" = "1" ]; then echo "RAM CRITICAL ${ram} GB -- inside the band where all 3 hangs occurred"; exit 2; fi
    if [ "$hi56" = "1" ] && [ "$warned56" = "0" ]; then echo "RAM 56 GB reached (${ram}) -- hang band entered"; warned56=1; fi
    if [ "$hi52" = "1" ] && [ "$warned52" = "0" ]; then echo "RAM 52 GB reached (${ram}) -- approaching the hang band"; warned52=1; fi
  fi
  sleep 60
done
echo "ram watch finished, last=${ram} GB"
