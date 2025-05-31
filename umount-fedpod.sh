#!/bin/bash
for i in {1..6}; do HOSTNAME=$(printf 'node%02d' $i); echo $HOSTNAME; ssh dwngcp2504@${HOSTNAME} 'sudo mkdir -p /fedpod && sudo umount /fedpod && ls /fedpod'; done;
