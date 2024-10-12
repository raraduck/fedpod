#!/bin/bash

echo -e "\033[32mSTART TEST: AGGREGATION\033[0m" && \
docker-compose -f compose-aggregation.yaml up && \
echo -e "\033[32mCOMPLETED: AGGREGATION\033[0m" && \
echo -e "\033[33m--------------------------------\033[0m" && \
echo -e "\033[32mSTART TEST: VALIDATION AND TRAIN\033[0m" && \
docker-compose -f compose-train.yaml up && \
echo -e "\033[32mCOMPLETED: VALIDATION AND TRAIN\033[0m"
