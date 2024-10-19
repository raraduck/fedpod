#!/bin/bash
echo -e "\033[32mSTART TEST: TRAIN\033[0m" && \
docker-compose -f compose-train_trn.yaml up && \
echo -e "\033[32mCOMPLETED: TRAIN\033[0m"