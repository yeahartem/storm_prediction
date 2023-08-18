#!/bin/bash
search_dir=test
for entry in "$search_dir"/*
do
  python -m unittest $entry
done
