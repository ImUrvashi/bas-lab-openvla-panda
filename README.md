# Language-driven Panda robot in an Unreal space station (Llama 3 + OpenVLA)

A Franka Panda arm in a space-station lab built in **Unreal Engine 5.8** performs five station tasks from a typed
request. **Llama 3 8B** turns the request into a task. **OpenVLA-7B**, fine-tuned on Unreal demonstrations, drives the arm
from one camera image. The model runs on a remote GPU server and is connected to Unreal in real time.

```
typed request ──► Llama 3 (task 1-5 or refuse) ──► Unreal sends camera frame + task sentence
      ▲                                                         │
      │                          gateway (Mac): image → PNG ──► OpenVLA (GPU, SSH tunnel)
      └── arm moves ◄── joint targets ◄── IK ◄── 7-D hand action [dx dy dz droll dpitch dyaw grip]
```

## Tasks

| # | Example request | Sentence sent to OpenVLA | Success when |
|---|---|---|---|
| 1 | "move the red block onto the yellow pad" | Pick up the red cube and place it on the yellow marker. | cube on marker, released |
| 2 | "stow the wrench and shut the drawer" | Open the drawer. → Put the tool in the drawer. → Close the drawer. | tool in drawer, drawer shut |
| 3 | "shut off the coolant" | Close the coolant valve. | lever turned 90° |
| 4 | "swap in the new filter" | Insert the filter cartridge into the air scrubber. | cartridge seated |
| 5 | "store the specimen tube in the container and close it" | Put the sample vial in the container. → Close the lid. | vial inside, lid latched |

OpenVLA only knows the sentences it was trained on. Llama picks the task, and Unreal sends the fixed sentence.
Tasks 2 and 5 switch sentence on state changes, because the start and end of the task look identical to the camera.

## Models

| | OpenVLA (action) | Llama 3 (front end) |
|---|---|---|
| Base | `openvla/openvla-7b`: DINOv2 + SigLIP vision, Llama 2 7B | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Adaptation | LoRA r32 / α16 on all linear layers: 110.8 M trainable (1.45 %). Output head and embeddings frozen. | none: system prompt + few-shot examples |
| Output | 7 action tokens (256 bins each) → un-normalised hand delta | task 0-5 + a one-line reply |
| Served as | bf16, unmerged LoRA; task 1 routed to its own adapter | 4-bit |

## Data and training

- **Demos:** 1,000 scripted-expert episodes (200 per task, all successful) recorded in Unreal, with DART noise so the model
  learns to recover. 15 GB raw.
- **Export:** every 3rd frame → **67,900 frames** from 950 episodes; 50 episodes held out (10 per task).
- **Training:** QLoRA (4-bit NF4 base), AdamW, batch 4 × 2 accumulation, 100 warm-up steps, cosine decay.

| Stage | Data | Steps | LR | Wall time | Used for |
|---|---|---|---|---|---|
| `side_v12` | MuJoCo lab, task 1 | 2,500 | 5e-4 | earlier work | start point |
| `t1_lr5e-4` | Unreal, task 1 | 2,500 | 5e-4 | — | start point |
| `bench_9k` | Unreal, 5 tasks | 9,000 | 5e-4 | ≈ 7.5 h | **task 1** (step 7,000) |
| `bench_sub` | same, sub-step sentences | 3,500 | 3e-4 | ≈ 3.7 h | **tasks 2-5** (step 3,500) |

## Results

**Live, in Unreal** (one run per task, OpenVLA driving, fixed layout):

| Task | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Result | ✅ | ✅ | ✅ | ✅ | ✅ |
| Steps | 90 | 228 | 83 | 91 | 137 |

**Offline, held-out frames** (320 frames, 40 per sub-step, compared with the expert):

| Serving | Move error | Direction cos | Gripper agree | Model time / step |
|---|---|---|---|---|
| 4-bit | 0.24 mm | 0.984 | 99.1 % | 0.97 s |
| **bf16 (used)** | 0.29 mm | 0.973 | 99.4 % | 0.55 s |
| bf16 merged | 0.29 mm | 0.976 | 99.1 % | 0.38 s, **rejected:** froze after grasping in task 4 |

**Llama 3 front end:** 32 / 35 (91 %) on a 35-request test set: direct 5/5, paraphrase 15/15, refuse 6/6,
typo 4/5, ambiguous 2/4. Mean latency 1.1 s.

**What moved the numbers:**
- Task 2 never finished with one sentence. The arm opened the drawer, then stopped, because "hand over closed drawer" is
  both the start and the end. Sub-step sentences fixed it.
- The sub-step model put the cube back in task 1, so task 1 is routed to the step-7,000 adapter.
- An object hanging 4 cm below the hand made the gripper open and close repeatedly. Snapping grasped objects to the
  grasp centre fixed it.
- An IK elbow-limit stall in task 5 was fixed with a nullspace pull towards the home posture.

## Hardware and timing

| | |
|---|---|
| Simulator | MacBook Air M5, 24 GB, Unreal Engine 5.8.4 (no CUDA) |
| GPU server | 8 × RTX A6000 48 GB, 96 cores, 250 GB RAM. PBS scheduler, max 2 jobs per user, reached over VPN. |
| Training memory | one A6000 (4-bit base) |
| Serving memory | ≈ 22 GB on one A6000 (OpenVLA bf16 + Llama 4-bit) |
| Recording | ≈ 15 h wall time for 1,000 demos |
| Upload | 10.3 GB of images → 1.6 GB with zstd, 8 parallel streams (VPN ≈ 150 KB/s per stream) |
| Control loop | 0.6–1.1 s per step (model 0.5–1.0 s + network). A task takes about 1.5–4 min. |
| Server start | 1 min (idle server) to 15 min (busy server) |

## Run the demo

Four terminals, from the repository root (Python env in `.venv`):

```bash
# 1. model server on the GPU (OpenVLA + Llama); prints SERVER READY
./server/start_demo_server.sh
# 2. tunnel to the server
ssh -N -L 8001:127.0.0.1:8001 -J <user>@<master> <user>@compute
# 3. OpenVLA gateway for Unreal (port 8000)
.venv/bin/python policy_ik_gateway.py --upstream-url http://127.0.0.1:8001/act --speed 2.5 --fine-speed 1.5
# 4. optional scripted fallback (port 8002), still uses Llama
.venv/bin/python policy_ik_gateway.py --policy expert --port 8002 --llm-url http://127.0.0.1:8001/interpret
```

In Unreal, open `unreal/BASPanda/BASPanda.uproject`, press **Play** and type into the box. Settings on the policy bridge actor:
- **Use Model:** OpenVLA, or the scripted expert when off.
- **Smooth Motion:** fetch the next action while the arm moves.
- **Substep Instructions**, **Use Language Front End**.
- **Randomize Layout:** objects shift up to ±1 cm per run, the trained range.

Add `--trace doc/trace.csv` to the gateway to log every step.

## Repository

| Path | What |
|---|---|
| `unreal/BASPanda/Source/BASPanda/` | Panda actor, workcell (5 tasks, grasping, state), policy camera, bridge + command box |
| `policy_ik_gateway.py` | Unreal ⇄ model: resize, PNG, IK, speed/smoothing options, scripted experts, recording, trace |
| `policy_server.py`, `llm_frontend.py` | GPU server: OpenVLA `/act` (adapter routing, bf16) and Llama 3 `/interpret` |
| `multitask_expert.py`, `expert_policy.py` | scripted experts for tasks 2-5 and task 1 |
| `build_unreal_export.py`, `finetune_panda.py`, `eval_checkpoints.py` | dataset export, LoRA training, offline evaluation |
| `openvla_adapter.py` | Panda FK/IK, action encode/decode |
| `server/` | PBS jobs (`train`, `serve`, `eval`) and `start_demo_server.sh` |
| `bas_lab/`, `closed_loop_panda.py`, … | original MuJoCo lab and its pipeline (below) |

## Limitations

- **Narrow generalisation.** One scene, one camera, objects within ±1 cm of their home spots. Larger moves, new objects
  or a new camera view need new demos.
- **Simplified contact.** Grasped objects snap to the gripper. Drawer, valve and lid move kinematically. Gravity is off
  (microgravity).
- **Slow control.** About 1 action per second versus 10 for the scripted expert. Action chunking (OpenVLA-OFT) and a
  dedicated GPU close to the simulator are the next steps.

## Origin: MuJoCo BAS lab

The project started from a MuJoCo version of the lab (task 1 only, gravity 0). It's described in `bas_lab/README.md`
and runs with `closed_loop_panda.py`. Its best model (`weights/side_v12_step2500/`, Git LFS, run `git lfs pull`)
placed the cube in 12/12 held-out and 12/12 fresh layouts. It is the starting point of the Unreal training above.
Panda model: MuJoCo Menagerie (Apache-2.0).
