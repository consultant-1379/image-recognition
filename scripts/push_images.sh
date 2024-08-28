#!/usr/bin/env bash

set -x
SCRIPT=$(readlink -f $0)
SCRIPTPATH=$(dirname $SCRIPT)
REPOROOT=$(dirname $SCRIPTPATH)

source ${SCRIPTPATH}/base.sh 

docker pull "${image}:${tag}"

for extratag in "${extra_tags[@]}"
do 
    echo "Pushing ${image}:${extratag}"
    docker tag "${image}:${tag}" "${image}:${extratag}" 
    docker push "${image}:${extratag}" 
done