#!/usr/bin/env bash

source ./download.sh
git pull
poetry run python -m app
