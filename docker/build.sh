#!/bin/bash

rm -rf src
cp -R ../src src

export IMAGE_NAME=open-land-use
docker build --rm=true -t $IMAGE_NAME .
docker tag $IMAGE_NAME registry.test.euxdat.eu/euxdat/$IMAGE_NAME
docker push registry.test.euxdat.eu/euxdat/$IMAGE_NAME

