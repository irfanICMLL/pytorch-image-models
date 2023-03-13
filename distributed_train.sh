#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC train.py "$@"

