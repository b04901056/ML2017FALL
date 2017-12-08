#!/bin/bash
python util.py $1
python w2v.py final_label.npy label.npy