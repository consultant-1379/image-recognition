#!/usr/bin/env bash 

SCRIPT=$(readlink -f $0)
SCRIPTPATH=$(dirname $SCRIPT)
REPOROOT=$(dirname $SCRIPTPATH)

source "${SCRIPTPATH}/base.sh"

s2i build models/inception3/  "${s2iBuilderImage}"  "${image}:${tag}" --pull-policy always

docker run -p 6000:6000 -p 9000:9000 -it  "${image}:${tag}"