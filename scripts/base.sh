#!/usr/bin/env bash 

major_version="3.1"
image="armdocker.rnd.ericsson.se/proj-mxe-models/image/img_inception3"
tag="${major_version}.1"
s2iBuilderImage="armdocker.rnd.ericsson.se/proj-mxe/seldonio/seldon-core-s2i-python37:1.9.0-01-ubuntu-20210827"

extra_tags=("${major_version}.2", "${major_version}.3" "${major_version}.4" "${major_version}.5" "${major_version}.6")