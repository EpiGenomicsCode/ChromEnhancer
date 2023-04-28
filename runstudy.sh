#!/bin/bash
# $1: study name
rm -rf output/ *out* *log*; pkill -9 python; clear; nohup python main.py --$1 > 1log.out & sleep 2 ;nohup python main.py --$1 >> 2log.out &  sleep 2 ;nohup python main.py --$1 >> 3log.out &