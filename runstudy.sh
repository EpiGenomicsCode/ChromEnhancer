#!/bin/bash
# $1: study name
rm -rf output/ *out; pkill -9 python; clear; nohup python main.py --$1 &  