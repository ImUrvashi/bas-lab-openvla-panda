#!/bin/bash
# Run on the Mac: start the demo model server (OpenVLA bf16 + task-1 route + Llama 3) on the least-used GPU,
# wait until it answers, then print its health. Needs the SSH tunnel for port 8001 to check from here.
set -e
R=/data/urvashib/openvla_panda
SUB=runs/openvla-7b+panda_pilot+b8+lr-0.0003+lora-r32+q-4bit--bench_sub/checkpoints/step_03500
T1=$R/runs/openvla-7b+panda_pilot+b8+lr-0.0005+lora-r32+q-4bit--bench_9k/checkpoints/step_07000/lora_adapter
GPU=$(ssh -J <user>@<master> <user>@compute \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1" \
    2>/dev/null | tail -1)
echo "starting on GPU $GPU"
ssh <user>@<master> "cd $R/code/server && qsub -v GPU=$GPU,PORT=8001,BF16=1,MERGE=1,LLM=meta-llama/Meta-Llama-3-8B-Instruct,ADAPTER=$SUB,ROUTE='pick up the red cube=$T1' serve.pbs" 2>&1 | grep -v modulefiles
echo "loading (about 5 minutes) ..."
until curl -s -m 5 127.0.0.1:8001/health 2>/dev/null | grep -q '"bf16": true'; do sleep 15; done
curl -s 127.0.0.1:8001/health; echo
echo "SERVER READY"
