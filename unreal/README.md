# BAS Panda Unreal work

Unreal Engine version: 5.8.4. The C++ project is `unreal/BASPanda/BASPanda.uproject`.

## Robot visual export

From the repository root:

```bash
python3 unreal/tools/export_panda_visuals.py
```

This writes `unreal/exported/Panda/`: eleven `.obj` files (one per rigid body), `PandaMaterials.mtl`, and `panda_manifest.json`. The OBJ vertices are in centimetres and retain MuJoCo's X/Y/Z coordinates. Unreal 5.8's Interchange OBJ importer reflects Y during import, so the resulting Static Mesh assets use Unreal X/-Y/Z coordinates. Each OBJ pivot is its link's MJCF body origin. The manifest contains parent transforms, nine joints and limits, material colours, home pose, and references to the original collision geometry. Exported files are reproducible and Git-ignored; keep the source XML, original meshes and exporter tracked.

The project has an **Empty Level** saved as `/Game/BAS_Lab`. The template landscape visible in the initial Untitled level is not the station reference scene.

The eleven OBJs have been imported as **Static Meshes** into `/Game` in the Unreal Content Browser. The actor currently loads them from that exact path. The `link0` bounds are approximately X -15.41 to 7.15 cm, Y -9.46 to 9.46 cm, Z 0 to 14 cm. Keep import scale at 1, since the files are already in centimetres. Do not combine the eleven bodies into a single static mesh: each link must move separately. The exporter emits one OBJ object per body and valid planar UVs for Unreal's Interchange importer.

The project now has `APandaRobotActor` in `Source/BASPanda/PandaRobotActor.*`. It assembles the 11 imported Static Meshes at the MuJoCo home pose with seven movable arm frames and two finger sliders. The C++ module compiled successfully on the installed UE 5.8.4 editor. Restart the Editor after the external build, place `PandaRobotActor` at world origin in `BAS_Lab`, switch the perspective viewport to **Unlit** if the empty level is dark, and focus the actor with **F**. In Details, editing `ArmJointsRad` or `FingerOpeningCm` updates the preview; `ResetHome` restores the source pose. `InspectImportedLinks` in the actor's `Panda|Diagnostics` section shows each loaded mesh's local and world Z bounds and visibility. If the actor was assembled from OBJs exported before the Y-axis fix, regenerate the files and reimport all 11 mesh assets; there is no need to recreate the actor or level.

This actor is currently a **kinematic visual assembly**: it has no robot physics constraints, collision bodies, servo drives, or runtime IK. It now exposes a hand-mounted `GraspFrame` and world grasp-centre query for the first task prototype. Next, validate its home hand pose and individual joint sweeps against MuJoCo, then build contact-capable articulation before connecting OpenVLA. Unreal's Python editor scripts are useful for import/setup but cannot run as gameplay control; the runtime controller will be C++.

For a first runtime command check, select `PandaRobotActor`, enable **Panda → Motion Test → Run Motion Test On Begin Play**, and start **Simulate** from the editor Play menu. Joint 1 moves at most 0.1 radians for two seconds and returns to its starting angle. Stop simulation, turn the checkbox off, then save the level. This checks runtime updates to the kinematic actor; it does not decode OpenVLA output or simulate robot joint physics.

For the next connection check, restart the mock server and Unreal Editor after rebuilding C++. Turn off the Panda motion test. Select `BASPolicyBridgeActor` and enable **BAS Policy Test → Apply Joint Targets**; leave **Send On Begin Play** enabled. Start **Simulate**. The bridge sends the current joints with the RGB observation, and the mock server returns a separate eight-value `joint_targets` field with joint 1 advanced by 0.1 rad. The Panda moves smoothly toward it. Stop simulation and disable **Apply Joint Targets**. The bridge rejects missing or invalid targets; the Panda clamps each step to 0.15 rad, applies source joint limits, and limits speed to 0.5 rad/s. The normal seven-value `action` remains unapplied. This proves one network command can move the kinematic arm; it does not perform IK, continuous control, grasping, or OpenVLA inference.

`policy_ik_gateway.py` is the next one-shot test. Create the ignored local environment with `python3 -m venv .venv` and `.venv/bin/python -m pip install mujoco numpy`, then run `.venv/bin/python policy_ik_gateway.py --port 8000` after stopping the mock server. Restart Unreal after rebuilding C++, select the bridge actor, enable **Apply Joint Targets**, and start **Simulate** once. In sample mode the gateway applies a fixed 3 cm hand delta through `openvla_adapter.decode_action()` and returns bounded joint targets. No OpenVLA weights are loaded in this mode. `--upstream-url http://127.0.0.1:8001/act` will later forward the image and instruction to a separately running `policy_server.py` and decode its seven-value action. The current server requires CUDA and missing model weights; do not set the upstream option until an inference endpoint passes its own test. The gateway is one-shot and does not yet implement a 10 Hz closed loop or collision avoidance.

## First station workcell preview

`ABASWorkcellActor` in `Source/BASPanda/BASWorkcellActor.*` is a source-measured workcell preview. It uses engine cube primitives for the robot mount, bench rack, bench top, red pick cube, and yellow target marker, plus a local point light and exposure adjustment. Place one actor at world `(0,0,0)` beside `PandaRobotActor` at `(0,0,0)`. The bench top is centred at `(55,0,35)` cm and reaches Z=37.5 cm; the cube starts at `(45,0,40)` cm; the target is centred at `(60,0,37.65)` cm. These values come from `bas_lab/mod_workstation.xml`. At runtime the cube simulates as an 80 g body with gravity disabled; `ResetCube` returns it to its source pose. A first-pass grasp proxy finds the Panda actor, attaches the cube when its centre is within 6 cm of the gripper centre and finger opening is at most 2.7 cm, and releases it when opening reaches 3.2 cm. `bCubeHeld` and `IsCubePlaced()` expose task state. This attachment is a task prototype, not physical finger contact; it has not been validated in an Unreal play session yet. The bench does not implement MuJoCo's adhesion actuator.

`ABASStationModuleActor` in `Source/BASPanda/BASStationModuleActor.*` adds the source-sized 24-panel cylindrical shell: 7 m long, 1.925 m panel-centre radius, centred at `(55,0,85)` cm. It also uses 1,350 box instances from 77 equipment racks and details in `bas_lab/mod_bas_module.xml`, plus circular end caps and hatch discs. `unreal/tools/export_bas_station.py` deterministically regenerates `StationModuleBoxes.inl` from the MuJoCo XML. Place the actor at `(0,0,0)` in the same level. `bCutawayForEditing` defaults to true and removes the +X side so the -X wall remains behind the side policy camera's view; turn it off in Details to show all 24 shell panels and rack sides. The station is visual only; texture maps, porthole detail, and physical contact are still pending.

`ABASPolicyCameraActor` in `Source/BASPanda/BASPolicyCameraActor.*` implements the **side free camera** used by `closed_loop_panda.py` for the best lab model, not the older `policy_cam` in `bas_lab/mod_cameras.xml`. Place it at world `(0,0,0)` after restarting the editor, then select it and use the viewport **Perspective → Pilot → BASPolicyCameraActor** menu to inspect the 4:3 frame. Its MuJoCo look-at is `(0.50, 0, 0.46)` m, distance 0.75 m, azimuth 90 degrees, elevation -15 degrees; Unreal position is approximately `(122.44, 0, 65.41)` cm and horizontal FOV 57.82 degrees (45 degrees vertical at 4:3). A camera-side point light without shadows provides a first approximation of MuJoCo's ambient headlight in the enclosed lab. `ObservationCapture` can later be assigned a 320x240 render target. Camera placement and lighting need a paired MuJoCo/Unreal image check before policy use. No image bridge or policy runtime is connected yet.

## OpenVLA integration order

### Local no-CUDA connection check

The repository includes `mock_policy_server.py`, which uses only Python's standard library and serves the same `/health` and `/act` JSON endpoints as `policy_server.py`. Its seven-value action is zero hand motion and fully open gripper. It never loads OpenVLA. When Unreal supplies joint state, it also returns a separate eight-value `joint_targets` test command. Unreal only applies that command when **Apply Joint Targets** is enabled. This is a transport and runtime-command check, not a task-success test.

1. In Terminal, from the repository root, run `python3 mock_policy_server.py --port 8000`. Keep that Terminal open. A ready line should contain `MOCK_POLICY_READY http://127.0.0.1:8000`. Opening `http://127.0.0.1:8000/` or `/health` in a browser shows status; `/act` requires a POST from Unreal.
2. Restart Unreal Editor after the C++ build, so it loads the new `ABASPolicyBridgeActor` class. Open the existing `BASPanda` project and `BAS_Lab` level. Keep the existing `BASPolicyCameraActor` in the Outliner.
3. Use **Place Actors** search for `BAS` and find `BAS Policy Bridge Actor`, then drag one into the level. Select it. In Details, keep `PolicyUrl` as `http://127.0.0.1:8000/act` and `Instruction` as `pick up the red cube`. If it is absent, fully quit Unreal and reopen the `.uproject`; the project module manifest in `Binaries/Mac/UnrealEditor.modules` must point to the newest compiled `libUnrealEditor-BASPanda` library. The build that introduced the bridge produced `...0013.dylib`.
4. Click `SendTestObservation` in the bridge actor's **BAS Policy Test** section. Alternatively press **Play** once; `bSendOnBeginPlay` sends one request. No need to click the red cube. The mock server Terminal should print `request 1: pick up the red cube (150528 RGB bytes)`.
5. In Unreal, select the bridge actor again and inspect `LastResult`. Success reads `OK: received seven values; gripper=1.00 (not applied)`. The server writes the last received image to the Git-ignored `doc/mock_policy_observation.png`. Open it and check that the Panda, cube, target marker and bench are visible. If it says `No BASPolicyCameraActor`, add the camera actor to the level. If it says `HTTP connection failed`, check the server Terminal and port. If it says `Scene capture readback failed`, check the Editor Output Log for capture errors. Save the level once the test succeeds.

The camera actor creates a transient 224x224 RGBA8 render target for this test if none is assigned, captures one frame and sends its raw RGB bytes as base64. The first capture/readback runs on Unreal's game thread and may briefly pause the editor; continuous policy control still needs an asynchronous or pipelined capture path. A server reply confirms the bytes arrived, but does not prove that the image is well framed. The next step is to inspect an actual captured image and compare it with MuJoCo before connecting an IK/action controller.

The existing `policy_server.py` serves `POST /act` on localhost and returns **seven** values: world-frame hand translation delta (metres), rotation delta (radians), and normalized gripper openness. `openvla_adapter.decode_action()` converts that output, with the current seven joint angles, into **eight** actuator targets: seven Panda joint angles (radians) and gripper openness (0 to 1). `closed_loop_panda.py` uses 320x240 camera frames resized to 224x224 and a 10 Hz policy loop. Keep these conventions in the Unreal bridge; do not send the server's seven outputs directly to `ArmJointsRad`.

1. **Validate the scene:** compare the Unreal home hand pose, joint sweeps, and policy camera image with the MuJoCo lab scene. Match the visible cube and target placement before using model output.
2. **Make the task physical:** validate the first-pass grasp proxy and placement check, then give the Panda contact-capable fingers and joint drives and reproduce the bench's zero-gravity adhesion behavior. The cube simulates, but the current Panda actor is still a kinematic visual assembly.
3. **Build a local observation/action bridge:** capture RGB at 320x240, resize to 224x224, send the exact `instruction`, `image_b64`, and `shape` fields to `policy_server.py`, then run the repo's IK decoder on the returned seven values. Clamp each joint step to 0.15 rad and the requested hand translation to 0.03 m per 10 Hz step, as in `closed_loop_panda.py`. Apply the resulting eight targets through a bounded runtime Panda controller. Keep HTTP/model inference off Unreal's game thread.
4. **Validate without the model first:** replay one recorded action sequence through the bridge, reset, and check that the cube can be picked up, moved, released near the marker, and remains there. Then run `policy_server.py` on a suitable GPU host and test the same instruction from fresh resets. Record image, action, cube position, and success for each step.

The installed Mac editor can build and preview the Unreal scene, and the mock connection test above runs entirely on this Mac without CUDA. The current `policy_server.py` hard-codes CUDA, so real model inference needs a separately adapted Apple MPS backend or a compatible remote GPU host. The repo does not contain the trained LoRA weights or OpenVLA base weights. Its existing endpoint can be accessed through a localhost port forward when the server is remote.

Source robot license: `mujoco_menagerie/franka_emika_panda/LICENSE` (Apache-2.0).

## Closed-loop episodes (scripted expert, no model)

`ABASPolicyBridgeActor` can run whole episodes: capture → `POST /act` → IK → one 0.1 s arm step, repeated. The world waits for each reply, so slow inference only slows playback. Each request also carries the cube position and marker XY (MuJoCo metres), used only by the expert.

1. Run `.venv/bin/python policy_ik_gateway.py --policy expert`.
2. On `BASPolicyBridgeActor`, tick **BAS Policy Loop → Run Closed Loop On Begin Play**. Optionally set **Num Episodes** and **Randomize Layout**.
3. Start **Simulate**. Read the `BAS loop:` lines in the Output Log: one per episode (SUCCESS/FAILED), then `Done: N/M successful` and the mean request latency.

The success rule matches `closed_loop_panda.py`: the cube is released on the marker (within 4 cm, at rest height) with gripper openness above 0.8, for 10 consecutive steps. While the cube is held, the fingers stop at 2.5 cm, so the measured openness is about 0.62, as in MuJoCo. For OpenVLA, start the gateway with `--upstream-url` instead of `--policy expert`.

## Recording task-1 demonstrations

1. Run `.venv/bin/python policy_ik_gateway.py --policy expert --record demonstrations_unreal --seed 1`. Use a new `--seed` for each session; it names the manifest and drives the noise.
2. On `BASPolicyBridgeActor`: tick **Run Closed Loop On Begin Play** and **Randomize Layout**, set **Num Episodes** (for example 50), and give **Layout Seed** a new value for each session so layouts don't repeat. Keep Unreal in focus, or turn off *Use Less CPU when in Background*.
3. Start **Simulate**. The gateway prints `SAVED expert_NNNN.npz` or `REJECTED ...` for each episode.
4. Build the training export: `.venv/bin/python build_unreal_export.py --out panda_openvla_data_unreal_t1`, then copy that folder to the GPU server.

The recipe is the same as `record_expert_demos.py`: correlated noise on the executed motion, with the clean expert action stored as the label, and 10% of episodes without noise. When the gateway sends phases, the bridge ends an episode only after the expert reaches DONE, so the recordings include the retreat. `demonstrations*/` and `panda_openvla_data*/` are Git-ignored.

## Five station tasks

All props are always on one station bench, each at a fixed home spot; the policy camera (`LookAtM`, `DistanceM`, `ElevationDeg`) frames the whole bench. `BASWorkcellActor` → **Task** only picks the instruction, success check and (for recording) the expert. Any loose object can be grasped and any handle gripped, so a wrong choice by the policy really happens.
Mechanisms are kinematic: a gripped handle drags its drawer, turns with the wrist (valve) or swings on its hinge (lid);
an item released inside the drawer/container, or a cartridge released aligned over the slot, snaps in and moves with it.

| Task | Props | Instruction | Success |
|---|---|---|---|
| 1 | red cube, yellow marker | Pick up the red cube and place it on the yellow marker. | cube on marker, released |
| 2 | cabinet with drawer (orange front, dark knob), green tool | Put the tool in the drawer and close it. | tool in drawer, drawer shut, hand free |
| 3 | coolant pipe, blue valve body, red quarter-turn lever, light | Close the coolant valve. | lever turned 90° (light green), hand free |
| 4 | low air scrubber with yellow-rimmed slot and light, teal cartridge | Insert the filter cartridge into the air scrubber. | cartridge seated (light green), hand free |
| 5 | purple container with hinged lid and latch light, pink vial | Put the sample vial in the container and close the lid. | vial inside, lid latched (light green), hand free |

Homes (MuJoCo metres): loose items in a row at x 0.55 in front of the arm (cube y −0.125, tool −0.04, vial 0.04, cartridge 0.12); cabinet (0.44, −0.27), valve (0.45, 0.25), pad (0.66, −0.17), scrubber (0.66, 0.02), container (0.625, 0.225). With **Randomize Layout**
on the bridge, each episode shifts every prop by up to `LayoutJitterM` (1 cm) around its home. The bridge sends the task's instruction (turn off
**Use Task Instruction** to send its own), plus `task`, `cube_pos` (active object), `target_pos` and a `state` object
(object/target/handle positions, drawer_open, valve_deg, lid_deg, tool_stowed, vial_stowed, seated, latched). Expert recording works for all five tasks (`multitask_expert.py` for tasks 2-5).

## Asking the robot (Play)

With Run Closed Loop on Begin Play off, press **Play** (not Simulate): a box at the bottom of the view takes a
request such as "close the coolant valve". Keywords pick the task (vial/container/lid, drawer/tool, valve,
filter/cartridge/scrubber, cube/marker); the trained sentence for that task is sent, and the task runs once.
**Mode** switches between two gateways:

    # OpenVLA (tunnel to the model server on 8001)
    .venv/bin/python policy_ik_gateway.py --upstream-url http://127.0.0.1:8001/act            # port 8000
    # scripted experts
    .venv/bin/python policy_ik_gateway.py --policy expert --port 8002
