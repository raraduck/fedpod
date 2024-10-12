#!/bin/bash
echo -e "\033[32mSTART TEST: VALIDATION\033[0m" && \
docker-compose -f compose-train_val.yaml up && \
echo -e "\033[32mCOMPLETED: VALIDATION\033[0m"
echo -e "\033[33m--------------------------------\033[0m" && \
echo -e "\033[32mSTART TEST: VALIDATION AND TRAIN\033[0m" && \
docker-compose -f compose-train_trn-val.yaml up && \
echo -e "\033[32mCOMPLETED: VALIDATION AND TRAIN\033[0m"
