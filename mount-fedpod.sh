#!/bin/bash
for i in {1..6}; do HOSTNAME=$(printf 'node%02d' $i); echo $HOSTNAME; ssh dwngcp2504@${HOSTNAME} 'sudo mkdir -p /fedpod && sudo mount -t nfs 10.128.0.6:/fedpod /fedpod && ls /fedpod'; done;
