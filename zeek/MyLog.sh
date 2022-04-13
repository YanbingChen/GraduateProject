#!/bin/bash

i=1;
start_time=$SECONDS
for pcap in "$@"
do
  echo "Processing - $i: $pcap";
  zeek -C -r "$pcap" MyLog
  cp conn.log ../out/zeek_logs/conn.log"$i"
  i=$((i+1))
done

elapsed=$(( SECONDS - start_time ))
echo "Elapsed time: $elapsed seconds"
