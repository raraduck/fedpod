awk -F '\'*,\'*' '{print $1 $2}' /fedpod/logs/fed_0/fed_metrics.csv
