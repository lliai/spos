#!/bin/bash
sh distributed_best_arch_train.sh 2 --total_epochs 240 --sched_type 'step'
sh distributed_best_arch_train.sh 2 --total_epochs 240 --sched_type 'cosine'
sh distributed_best_arch_train.sh 2 --total_epochs 360 --sched_type 'step'
sh distributed_best_arch_train.sh 2 --total_epochs 360 --sched_type 'cosine'
