# Reconfigurable Vision-Guided Robotic Microinjection Workstation for Supervised Protocol Development

## Short Abstract

Automated microinjection systems in the robotics literature commonly emphasize high-throughput autonomous batch injection using specialized microrobots, microfluidic handling, visual target recognition, and/or force sensing. In contrast, many experimental laboratories need a reconfigurable system for low-to-moderate throughput protocol development, where expert microscope judgement remains valuable but repetitive approach, retraction, washing, volume delivery, and evidence capture should be made repeatable. We developed a supervised robotic injection workstation around a Dorna 2 six-axis manipulator, dual UVC microscope views, an Intel RealSense stream, an Arduino-driven syringe/plunger axis, and a Python control station. The system combines operational-space joystick teleoperation, local-tool-frame insertion/retraction, named joint-space pose programs, midpoint-guarded routine execution, syringe endstop calibration, synchronized video/telemetry recording, configurable halt sensitivity, and measured motion characterization. The contribution is a traceable, reconfigurable robotic workflow for microscope-guided injection rather than a fully autonomous high-throughput injector.

## System Overview

The platform is controlled by `dorna_joy_control.py`, a threaded Python application that separates human input, camera acquisition, robot command dispatch, syringe actuation, and user interface rendering. The Dorna controller is reached over a JSON/WebSocket command interface at `10.42.0.11:443`, with startup settings stored in `.dorna_launcher.json`. The active configuration uses two UVC microscope cameras at `640 x 480 @ 30 fps`, an Intel RealSense stream at `1280 x 720 @ 30 fps`, and a serial syringe controller at `115200 baud`.

Manual control is provided through a pygame joystick polling thread running at 240 Hz. The controller applies a deadzone of 0.02 and supports manual speed scaling from 1% to 500% of nominal command scale. Named poses are stored as six-joint configurations in `poses.json`; the current library includes `Default`, `Reload`, `Injector_Pose`, `Safe_Pose`, `Wash1`, `Wash2`, and `Calibration`, with optional `__midway` poses that enforce staged transitions before final approach.

## Field Context and Novelty

The closest prior art demonstrates that microinjection can be fully automated with high throughput when the target class, sample geometry, and immobilization strategy are tightly controlled. Wang et al. reported automated zebrafish embryo injection at 15 embryos/min with high survival and success rates using computer vision, microrobots, and a dedicated holding device. More recent systems extend this direction with batch embryo/larva recognition, microfluidic sample handling, deep-learning target localization, and microforce perception.

This platform is positioned in a different part of the design space. It is meant for protocol development and heterogeneous microscope-guided injection tasks where the target, needle, material, and wash sequence may change between sessions. The novelty is the integration of:

- A six-DOF desktop arm operated in both joint space and local TCP task space.
- Dual-microscope plus RealSense visualization with robust Linux UVC enumeration and stream recovery.
- A pose/routine domain-specific language for repeatable supervised workflows.
- A calibrated serial syringe axis with endstop-aware wash and expel cycles.
- Per-step multimodal data provenance: robot telemetry, joystick state, three camera streams, UI video, and volume metadata.
- A launcher-based characterization routine that produces robotics-standard plots from measured joint feedback and forward kinematics.

The system should not be presented as outperforming high-throughput autonomous embryo injectors. Its more defensible contribution is lowering the barrier to repeatable, instrumented, operator-supervised robotic injection when full custom automation is not yet appropriate.

## Contribution Statement

**Research gap.** Manual microscope-guided injection is flexible but hard to reproduce; fully autonomous batch systems are reproducible but often require specialized fixtures, fixed sample classes, and substantial vision/handling infrastructure.

**Proposed approach.** A reconfigurable supervised robotic workstation that keeps expert visual judgement in the loop while moving the repetitive, stateful, and recordable parts of the procedure into software.

**Measurable output.** The platform records trajectories, TCP motion, velocity profiles, syringe calibration constants, routine phases, operator inputs, and camera evidence, enabling later comparison of protocols rather than relying only on operator notes.

**Current limitation.** Biological endpoint validation is still required: injection survival, deposited-volume accuracy, target localization error, wash carryover, and post-injection phenotype/outcome rates have not yet been quantified in this repository.

**Poster claim hierarchy.**

- **Demonstrated now:** instrumented robotic workflow, calibrated syringe actuation, routine execution, measured joint/TCP trajectory logging, and synchronized evidence capture.
- **Strong technical claim:** the system converts an operator-dependent microscope task into a programmable, characterizable, and auditable robot-assisted procedure.
- **Do not overclaim yet:** autonomous target recognition, biological success rate, quantitative injected-volume accuracy, and force-confirmed puncture are future validation items.

## Technical Methods

### Control Architecture

The robot is controlled as a hybrid supervisory/teleoperation system. A Tk startup launcher configures hardware and safety parameters, then the pygame control station runs a 120 Hz robot-control loop, a 240 Hz joystick polling thread, independent camera acquisition threads, and serial syringe/endstop polling. Shared state is protected by explicit locks, while robot commands are serialized through a queue to avoid concurrent writes to the Dorna WebSocket controller.

Manual teleoperation is implemented as an operational-space jog controller. Joystick axes are shaped with deadbanding, snap-to-zero release logic, and first-order low-pass filtering before being converted into incremental `lmove` or `jmove` commands. The left stick uses axis arbitration to reduce diagonal cross-coupling; the live command stream is rate-limited to 40 Hz for general TCP motion and 80 Hz for tool-axis translation. Orientation is represented internally on SO(3); axis-angle commands are converted to rotation matrices, composed in the tool or world frame, and re-orthonormalized to suppress numerical drift. Tool-axis advance/retract commands project translation along the current TCP z-axis, so insertion and retraction are expressed in the local needle frame rather than in a fixed world axis.

In control terms, the joystick generates a bounded Cartesian twist command `xi_cmd = [v_x, v_y, v_z, omega_x, omega_y, omega_z]^T`. Translational increments are applied in either the world frame or the local tool frame, while rotational increments are integrated by composing incremental rotations on SO(3). For needle insertion, the commanded displacement is `Delta p = Delta s R_TCP e_z`, so the approach vector follows the current tool axis even when the microscope-aligned needle is not parallel to the robot base axes. This is the main task-space control feature that makes the interface relevant to microinjection rather than generic Cartesian jogging.

For a measured motion characterization, the launcher now includes a **Motion Characterization** action. It connects to the robot, moves to `Default`, applies the selected halt threshold/duration, commands a short `Default -> Reload -> Default` sweep, and logs joint feedback, forward-kinematic TCP pose, velocity norms, Jacobian singular values, manipulability, and alarm state to `poster_assets/characterization/<timestamp>/motion_characterization.csv`.

The launcher also includes a **Repeatability Study** action for a more defensible repeatability figure. It repeats `Default <-> Reload` for five cycles and records the settled endpoint TCP and joint residual after every move to `poster_assets/repeatability/<timestamp>/pose_repeatability.csv`.

The launcher also includes a **Right-Stick TCP Demo** action. It emulates the manual right-stick orientation commands (`up`, `down`, `left`, `right`) by commanding small pitch/yaw changes while holding the current TCP `x-y-z` target fixed. The resulting log separates joint reconfiguration from residual TCP drift, making it a direct demonstration of orientation control at an approximately stationary tool tip.

### Robot Motion and Pose Management

The robot layer uses the Dorna Python API with the `dorna_ta` kinematic model. Motion primitives are sent as JSON commands, primarily absolute joint moves (`jmove`, `rel=0`) to stored six-axis configurations. The pose scheduler supports two-stage approach:

- First move to `<pose>__midway` when available.
- Wait for explicit operator confirmation or an `ADVANCE_AUTO` routine token.
- Advance to the final named pose.

This reduces risk during approach by separating gross positioning from final insertion or wash alignment. The active approach distance is `70.0 mm`, and the tool length setting is `30.0 mm` in the current `settings.json`.

The pose/routine scheduler can be described as a guarded hybrid automaton: continuous manual or robot motion is permitted inside each state, but transitions to final approach and fluid actuation are gated by explicit routine tokens, operator confirmation, or settle conditions. That framing makes the system easier to defend as a robotics control workflow, because the safety-relevant actions are not hidden inside an informal operator habit.

Forward kinematics are computed from the Dorna model as `x = f(q)`, where `q = [j0 ... j5]^T`. The characterization logger evaluates a central-difference geometric Jacobian `J(q) = partial f / partial q` at the configured TCP, the minimum singular value `sigma_min`, the condition number `kappa = sigma_max / sigma_min`, a normalized Yoshikawa-style six-dimensional manipulability index `w = sqrt(det(J_n J_n^T))`, and the translational dexterity `w_v = sqrt(det(J_v J_v^T))`. The translational rows of `J_n` are scaled from mm/rad to m/rad before the mixed translational/rotational metric is formed. For older logs that lack Jacobian fields, `generate_poster_assets.py` recomputes `J_v` offline from the measured joint feedback using the same tool-center configuration, so Figure 10 remains model-derived from measured motion rather than purely nominal.

### Vision Subsystem

Two non-RealSense UVC camera groups are discovered through `/dev/v4l/by-path`, deduplicated by USB topology, and filtered to exclude RealSense depth/infrared nodes. Capture threads use OpenCV with resilient startup and recovery logic:

- Candidate backends: `CAP_ANY` and `CAP_V4L2`.
- Candidate encodings: `YUYV`, `MJPG`, and backend default.
- Candidate frame rates: requested FPS, then 15 and 10 fps fallbacks.
- Buffer size request: one frame, minimizing visual latency.
- Automatic reopening after repeated frame failures.

The RealSense thread attempts color streaming first, then infrared, then colorized depth if prior modes fail. Frames are published through lock-protected latest-frame references to avoid unnecessary per-frame copying in the live UI.

### Syringe and Wash Control

The syringe/plunger axis is controlled by an Arduino serial protocol using velocity commands of the form `V<rate>` and endstop queries with `E`. The current plunger settings are:

- Forward rate: `1600` actuator units/s.
- Backward rate: `1600` actuator units/s.
- Nominal syringe volume: `10.0 uL`.
- Calibrated full travel: `13.8008 s`.
- Calibrated full travel units: `22081.3`.
- Injection step: `1.0 uL`, corresponding to `2208.1` actuator units and `1.3801 s`.

The wash routine executes backward-to-endstop and forward-to-endstop strokes, using endstop transition detection from either structured `L/R` serial states or configured keywords such as `ENDSTOP`, `LIMIT`, `LIM`, `HIT`, and `END`.

Xbox-to-Arduino control is implemented as a guarded velocity interface rather than a direct volume command to the microcontroller. The pygame joystick thread samples the controller at `240 Hz`, stores trigger axes and button states in shared state, and the main UI loop maps normalized trigger displacement to ASCII velocity commands. Outside injection mode, the left and right triggers provide proportional manual plunger jogging after pose-confirmation and calibration gates are satisfied. During injection mode, a right-trigger rising edge starts a calibrated dose step, computes the target actuator travel `s_step = s_full Q_step / Q_full`, integrates commanded actuator motion `s_k = s_(k-1) + |V_cmd| dt`, and sends `V0` when the target travel is reached. This is an open-loop calibrated displacement controller for the syringe axis; it should be distinguished from measured delivered-volume feedback.

### Routine Language

The routine engine parses a small domain-specific language from `routine.txt` or `routines.json`. Supported commands include:

- `POSE <name>` or `GO <name>` for named pose movement.
- `ADVANCE_WAIT` for operator-gated final approach.
- `ADVANCE_AUTO` for automatic final approach.
- `ADVANCE_CANCEL` to skip the next midway pose.
- `WASH N` for repeated syringe wash cycles.
- `PLUNGER FWD` and `PLUNGER BWD` for explicit endstop strokes.

The active default routine moves through injector, safe, wash, and safe poses, with wash cycles between `Wash1` and `Wash2`.

For the poster, this routine should be presented as a supervisory state machine rather than as a script. That framing is more robotics-specific: the routine defines guarded transitions, operator handoff points, and preconditions for fluid actuation.

### Safety and Monitoring

Safety is implemented at several levels. The Dorna controller halt settings are configured with a threshold of `30` and duration of `50`, substantially more sensitive than the stock `200 / 10000` preset. Startup can clear alarms, apply halt settings, and run a sequential auto-tune routine that searches for the most sensitive stable halt pair. The software also stores a self-collision envelope using per-link radii `[32, 35, 20, 32, 24, 18] mm`, an `8 mm` pair margin, a `75 mm` base radius, and a `230 mm` base height. Host thermal zones are polled, with warning and clear thresholds at `95 C` and `90 C`.

### Data Recording

For each injection step, the recorder can create a timestamped directory under `~/Desktop/RobotInjectionData`. The recorder writes:

- `telemetry.csv`: relative time, six joint values, joystick axes/buttons, step distance, and step volume.
- `frames.csv`: indexed frame timestamps for each camera stream.
- `rs.mp4`, `uvc1.mp4`, `uvc2.mp4`: camera videos when enabled.
- `ui.mp4`: full UI recording when enabled.

This allows later reconstruction of robot state, operator input, visual evidence, and injected volume for each experimental step.

The data-recording layer is central to the novelty claim because it makes each supervised intervention auditable. In a poster, this is stronger than describing the platform as a joystick-controlled arm: the robotic contribution is that subjective microscope manipulation is transformed into a time-aligned dataset.

## Figures

Use these generated figures in the poster:

- [Figure 1: Pose-transition burden matrix](poster_assets/pose_transition_matrix.svg)
- [Figure 2: Staged transition burden from Default](poster_assets/pose_transition_distance.svg)
- [Figure 3: Control and acquisition timing budget](poster_assets/system_architecture.svg)
- [Figure 4: Syringe operating envelope](poster_assets/syringe_calibration.svg)
- [Figure 5: Joint trajectory during characterization sweep](poster_assets/characterization_joint_trajectory.svg)
- [Figure 6: Joint velocity profile](poster_assets/characterization_joint_velocity.svg)
- [Figure 7: TCP task-space trajectory and return closure](poster_assets/characterization_tcp_path.svg)
- [Figure 8: Peak-normalized joint and TCP speed envelopes](poster_assets/characterization_speed_norms.svg)
- [Figure 9: Supervisory phase timeline](poster_assets/characterization_phase_timeline.svg)
- [Figure 10: Normalized translational dexterity along measured trajectory](poster_assets/characterization_manipulability.svg)
- [Figure 11: Field positioning and contribution boundary](poster_assets/field_positioning_matrix.svg)
- [Figure 12: Contribution stack](poster_assets/contribution_pipeline.svg)
- [Figure 13: Measured characterization summary](poster_assets/characterization_summary.svg)
- [Figure 14: Supervisory routine state machine](poster_assets/routine_state_machine.svg)
- [Figure 15: Multimodal evidence and provenance schema](poster_assets/data_provenance_schema.svg)
- [Figure 16: Settled endpoint repeatability](poster_assets/endpoint_repeatability.svg)
- [Figure 17: Joint target residual during pose transitions](poster_assets/target_tracking_residual.svg)
- [Figure 18: Local task-space reachability and translational manipulability](poster_assets/local_workspace_manipulability.svg)
- [Figure 19: Right-stick fixed-TCP response envelopes](poster_assets/right_stick_tcp_decoupling.svg)
- [Figure 20: Xbox-to-Arduino syringe control stack](poster_assets/xbox_arduino_control_stack.svg)

Figure audit: Figures 1-4, 18, and 20 are configuration/model-derived but decision-relevant rather than generic schematics; Figures 5-9, 13, 17, and 19 are measured launcher-study plots; Figure 10 is model-derived from measured joint feedback when logged Jacobian fields are absent; Figure 16 requires the launcher repeatability study; Figures 11, 12, 14, and 15 should be used as explanatory/context figures, not as primary evidence.

For a crowded scientific poster, prioritize Figures 1, 3, 4, 5, 6, 8, 10, 13, 16, 17, 18, 19, and 20. Use Figures 11, 12, and 15 only if there is space for context; they help explain the system but are weaker as experimental evidence.

The figures are regenerated by:

```bash
python3 generate_poster_assets.py
```

## Suggested Poster Layout

### Left Column: Motivation and System

Robotic microinjection requires repeatable tool positioning, high-magnification visual feedback, and controlled fluid delivery. Manual operation alone can introduce variability in approach path, dwell time, and injected volume. This workstation combines an affordable six-axis desktop manipulator with microscope-guided teleoperation and a calibrated syringe drive to improve repeatability while preserving operator supervision.

Place Figures 3, 11, 12, and 15 here to show the architecture, field positioning, contribution boundary, and evidence stream. This helps prevent the poster from being interpreted as a generic joystick robot.

### Middle Column: Control and Calibration

The control architecture uses a threaded design to prevent camera stalls, joystick input, or serial endstop polling from blocking UI responsiveness or robot command dispatch. Joint-space poses define reproducible task states; midpoint poses constrain final approach; the routine DSL formalizes repeated injection/wash workflows. Syringe calibration maps full-stroke travel time and actuator units onto requested volume, enabling volume-based step commands.

Place Figures 1, 4, 5, 6, 8, 13, 16, 17, 19, and 20 here.

### Right Column: Workflow and Quantitative Configuration

The current workflow begins with robot startup, alarm clearing, camera selection, optional halt tuning, pose selection, tool-center alignment, injection-step recording, and wash cycles. The existing pose library spans a large joint-space range relative to `Default`, with dedicated injector, safe, wash, reload, and calibration configurations. Present the measured characterization results as proof that the platform can report its own trajectory, timing, and speed envelopes.

Place Figures 2, 7, 9, 10, and 14 here, plus a photo or screenshot of the actual machine if available.

## Figure Captions

**Figure 1. Pose-transition burden matrix.** Each cell reports the joint-space transition burden from the row pose to the column pose, computed as `||q_dst - q_src||_2` over the six stored joint angles in degrees. The diagonal is zero because no reconfiguration is required, and darker/larger cells indicate transitions that require a larger coordinated motion of the arm. The blue `M` marker means the destination has a defined `<pose>__midway` posture, so the controller can split the move into gross approach and final approach rather than jumping directly to the target. In the current pose library, the largest burdens occur between the wash poses and the default/calibration/injector poses, mostly because `j4` changes by roughly `174-183 deg`; those are the transitions where midpoint gating is most justified. Small cells, such as `Default <-> Injector_Pose` at about `11 deg` and `Default <-> Calibration` at about `9 deg`, represent poses that already live in a similar arm configuration and are lower-risk direct moves. This figure should be read as a configuration-space risk and scheduling map, not as a Cartesian distance or collision certificate.

**Figure 2. Staged transition burden from Default.** Joint-space displacement, maximum single-axis motion, and midpoint availability for transitions out of `Default`. Larger values indicate higher reconfiguration effort and greater need for staged motion through midpoint poses.

**Figure 3. Control and acquisition timing budget.** Quantitative rates for joystick sampling, robot control, TCP jog command streams, microscope cameras, and RealSense acquisition. This makes the architecture figure less obvious by exposing the timing constraints of supervised teleoperation.

**Figure 4. Syringe operating envelope.** This figure maps requested injection volume to the syringe actuator command using the current endstop-derived calibration. The configured full stroke is `10.0 uL`, corresponding to `22081.3` actuator units and `13.8008 s` of travel; therefore a `1.0 uL` step corresponds to `2208.1` units and `1.3801 s`. The red calibration line is the assumed linear conversion from volume to actuator travel, while the side cards report practical operating quantities: full-stroke capacity, per-step resolution, total available steps, and the current remaining-volume register. In the present settings the remaining register is `0.0 uL`, so the figure indicates that the syringe state should be refilled, recalibrated, or reset before claiming another injection sequence. This is an operating-envelope figure rather than a delivered-volume validation figure: it documents the commanded fluidic model, but droplet weighing, imaging, or another assay is still required to quantify actual delivered-volume error, backlash, compliance, or clogging.

**Figure 5. Joint trajectory during characterization sweep.** Measured or nominal joint trajectories for the launcher characterization motion. With measured data, this plot reports controller feedback during `Default -> Reload -> Default`.

**Figure 6. Joint velocity profile.** Finite-difference joint velocities derived from the characterization log. This highlights whether the motion is dominated by one joint, whether velocity peaks are synchronized, and whether starts/stops are smooth.

**Figure 7. TCP task-space trajectory and return closure.** Forward-kinematic TCP trace in the world-frame `x-y` plane, with integrated path length, start/end closure error, horizontal envelope, and vertical span. This is stronger than a raw path trace because it quantifies whether the characterization sweep returns to its initial TCP.

**Figure 8. Peak-normalized joint and TCP speed envelopes.** This plot compares the timing of configuration-space speed and Cartesian tool speed during the measured characterization sweep. The purple curve is `||dq/dt||`, the Euclidean norm of the six joint velocities in `deg/s`; the blue curve is `||dx/dt||`, the translational TCP speed norm from forward kinematics in `mm/s`. Because those units are not physically comparable, each curve is normalized by its own peak and the actual peak magnitudes are reported in the side panel. In the current run, the maximum joint-speed norm is `33.02 deg/s` at `t = 8.32 s`, while the maximum TCP-speed norm is `60.62 mm/s` at `t = 10.32 s`, both during the `Default -> Reload` motion. The return move has slightly lower peaks (`32.51 deg/s` and `57.25 mm/s`). The figure is therefore a timing and coupling diagnostic: it shows when high joint activity produces high tool motion, whether starts/stops are smooth, and whether the commanded sweep creates unexpected task-space speed amplification.

**Figure 9. Supervisory phase timeline.** Timing of baseline hold and commanded motion phases, useful for reporting cycle time and separating dwell, transit, and settle periods.

**Figure 10. Normalized translational dexterity along measured trajectory.** Minimum singular value and translational manipulability `sqrt(det(J_v J_v^T))` along the measured joint trajectory. Low values identify configurations with reduced instantaneous Cartesian dexterity or increased sensitivity to joint perturbation.

**Figure 11. Field positioning and contribution boundary.** Qualitative comparison between manual micromanipulation, high-throughput autonomous batch injection, this supervised platform, and future validation targets. The figure clarifies that the contribution is reconfigurability and traceability rather than maximum throughput.

**Figure 12. Contribution stack.** End-to-end structure of the workstation: sensing, state estimation, control, protocol execution, and evidence capture. This is the most concise graphical statement of what is novel about the system integration.

**Figure 13. Measured characterization summary.** Quantitative dashboard for the latest launcher characterization run, including sample count, cycle duration, phase count, speed norms, TCP excursion, alarm state, and the source of dexterity metrics.

**Figure 14. Supervisory routine state machine.** Programmed routine represented as guarded robot states and fluidic actions. This makes the control contribution clearer than listing routine commands as text.

**Figure 15. Multimodal evidence and provenance schema.** Time-aligned data products created for each injection step, linking robot feedback, operator input, vision streams, syringe metadata, and output files.

**Figure 16. Settled endpoint repeatability.** Centered TCP endpoint scatter and summary statistics from repeated `Default <-> Reload` returns. This should be used once the launcher Repeatability Study has been run.

**Figure 17. Joint target residual during pose transitions.** Configuration-space residual between measured joint feedback and the active target pose during `Default -> Reload -> Default`. This is a standard control-style plot because it shows convergence toward each commanded endpoint, not only the raw trajectory.

**Figure 18. Local task-space reachability and translational manipulability.** Offline finite-difference forward-kinematic sweep around `Injector_Pose`, with color representing `sqrt(det(J_v J_v^T))`. This is a robotics-standard dexterity/workspace figure and should be presented as model-derived rather than experimentally measured.

**Figure 19. Right-stick fixed-TCP response envelopes.** Startup launcher demo for the right-stick `up/down/left/right` orientation controls. The robot is moved to `Default`, the base TCP position `p0` is recorded, and then small pitch/yaw orientation pulses are commanded while holding the target TCP `x-y-z` fixed. The figure now follows the Figure 8 style: it is a time-indexed response-envelope plot with shaded command intervals and quantitative cards at right. The purple curve reports joint reconfiguration `||q - q0||` normalized by its own peak, while the blue curve reports residual TCP drift `||p - p0||` relative to a `1 mm` translational reference. In the measured run, the maximum joint reconfiguration was `16.84 deg` and the maximum TCP drift was `0.351 mm`, so the TCP drift envelope remains far below the reference even during visible joint-space motion. This demonstrates that right-stick orientation control is not equivalent to translating the needle tip. It complements Figure 8: Figure 8 shows how joint and TCP speeds couple during a pose transition, while Figure 19 shows an orientation-only command where joints move but the TCP position remains approximately stationary.

**Figure 20. Xbox-to-Arduino syringe control stack.** Technical schematic of the syringe-axis control path from Xbox controller input to Arduino serial commands. The figure separates the `240 Hz` pygame joystick polling thread, trigger normalization/deadbanding, supervisory gates, velocity-command synthesis, and the `115200 baud` ASCII serial bridge. The proportional map shows how left and right trigger displacement is converted to signed `V<rate>` commands, while the dose-integrator panel shows the calibrated injection logic: `s_step = s_full Q_step / Q_full`, `s_k = s_(k-1) + |V_cmd| dt`, and `V0` at step completion. With the current calibration, a `1.0 uL` step corresponds to about `2208.1` actuator units and `1.380 s` at the configured rate. Endstop feedback is handled separately through `E` queries and parsed `ACT L/R`, `EL/ER`, or keyword responses during calibration and wash routines. This figure is useful because it makes clear that the Arduino is a velocity/endstop actuator interface, while the host computer owns dose accounting, workflow gating, and data provenance.

## Discussion Points

The main technical contribution is not a new manipulator design, and it should not be framed as a replacement for mature high-throughput autonomous microinjection robots. The contribution is a pragmatic robotics layer that makes a desktop six-axis arm usable as a traceable microscope-guided injection workstation. The software addresses constraints that appear repeatedly in real laboratory operation: unstable UVC enumeration, multiple camera modalities, staged approach, serial endstop uncertainty, repeatable wash cycles, calibrated volume delivery, and synchronized evidence capture.

This is pertinent to robotic biomanipulation because it targets a middle regime between manual micromanipulation and fully autonomous batch injection. That regime is important when protocols are still changing, the injected material or target morphology varies across experiments, or the investigator needs visual authority over the final puncture/injection decision. The current implementation is appropriate for operator-supervised research use; for unattended operation, the next technical priorities are quantitative injection-volume validation, camera-to-robot extrinsic calibration, image-based needle/target tracking, closed-loop visual servoing, force or pressure-based puncture-state detection, and formal collision checking against the full physical scene.

## Measured Characterization Results

The latest measured characterization log is `poster_assets/characterization/20260427_151253/motion_characterization.csv`. It contains `1013` samples over `25.85 s`, including a baseline hold at `Default`, a `Default -> Reload` move, and a `Reload -> Default` return move. No controller alarm was latched during the run.

Measured TCP excursion from forward kinematics was `69.66 mm` in `x`, `146.89 mm` in `y`, and `160.49 mm` in `z`. The integrated TCP arc length for the closed `Default -> Reload -> Default` sweep was `539.42 mm`, while the start/end TCP closure error was `0.095 mm`. The maximum measured configuration-space speed norm was `33.02 deg/s`, and the maximum TCP speed norm was `60.62 mm/s`. These values can be reported with Figures 5-10 as the motion characterization of the current pose library and controller settings.

The latest fixed-TCP right-stick demo log is `poster_assets/right_stick_demo/20260428_163025/right_stick_tcp_demo.csv`. It contains `812` samples from a `10 deg` orientation-pulse sequence at `40 Hz`, with commanded `up`, `down`, `left`, and `right` right-stick equivalents. The maximum joint reconfiguration from the base posture was `16.84 deg`, while the maximum TCP position drift was `0.351 mm`; baseline drift before the pulses was `0.097 mm`. Peak rates during the demo were `15.38 deg/s` in joint space and `7.20 mm/s` residual TCP motion. Figure 19 should therefore be presented as evidence of kinematic decoupling: right-stick orientation commands substantially change joint configuration while keeping the TCP position nearly stationary relative to the much larger task-space motion in Figure 8.

Use Figure 13 as the compact quantitative results panel. It is especially useful for a poster because it separates what has already been measured from what remains a planned biological validation experiment.

## Methods Summary

The system was implemented in Python using pygame for joystick/UI interaction, OpenCV for UVC capture and video writing, pyrealsense2 for RealSense streams, pyserial for syringe-axis communication, and the Dorna 2 Python API for robot commands. Configuration was serialized through JSON files for poses, routines, startup parameters, and calibration constants. Robot pose execution used absolute joint targets with optional midpoint gating, and injection recording synchronized telemetry and videos by relative timestamps.

## Characterization Protocol

For robotics-publication figures, run the launcher and press **Motion Characterization** before starting normal control. This produces a CSV log with columns for time, phase label, target pose, six joint angles, TCP pose, joint-speed norm, TCP-speed norm, singular values, Jacobian condition number, manipulability, and alarm state. Then regenerate figures:

```bash
python3 generate_poster_assets.py
```

Without a measured CSV, Figures 5, 6, 8, and 9 are generated from a nominal smooth interpolation between the stored `Default` and `Reload` poses, while Figures 7 and 10 explicitly indicate that measured characterization is required.

The current latest log was collected before Jacobian fields were populated, but the asset generator now backfills Figure 10 by applying offline finite-difference forward kinematics to the measured joint feedback. Future launcher logs include the Jacobian and joint-target residual columns directly.

For endpoint repeatability, press **Repeatability Study** in the startup launcher. That produces `poster_assets/repeatability/<timestamp>/pose_repeatability.csv`; rerun `python3 generate_poster_assets.py` to populate Figure 16.

For the fixed-TCP right-stick demonstration, press **Right-Stick TCP Demo** in the startup launcher. That produces `poster_assets/right_stick_demo/<timestamp>/right_stick_tcp_demo.csv`; rerun `python3 generate_poster_assets.py` to populate Figure 19.

Future motion-characterization logs now include `joint_target_error_norm_deg` and `joint_target_error_max_deg` directly. For older logs, Figure 17 reconstructs the same residual from `metadata.json` and the recorded target labels.

## Validation Plan

To make the poster stronger as a robotics contribution, the next experiment should quantify repeatability and biological relevance rather than only showing that the robot moves. Recommended metrics:

- **Tip placement repeatability:** return to the same microscope target after repeated `Default -> Injector_Pose -> Safe_Pose` cycles; report image-plane and reconstructed TCP error.
- **Injected volume accuracy:** command multiple 1 uL steps into a calibration target or droplet assay; report mean, standard deviation, and drift across the syringe stroke.
- **Protocol timing:** compare manual versus robot-assisted wash/injection sequence time for the same operator.
- **Operator intervention rate:** count how often the operator cancels a midway advance, pauses, or manually jogs during a scripted routine.
- **Biological endpoint:** for the intended model, report survival/viability, successful delivery fraction, and visible leakage or clogging events.
- **Safety envelope:** report alarm events, thermal warnings, and minimum self-collision margin during characterization runs.

## Threats to Validity

The current figures establish system capability and measured robot motion, but they do not yet prove biological performance. The main threats are uncalibrated camera-to-robot extrinsics, unknown image-plane-to-needle-tip error under the microscope, syringe compliance/backlash, possible needle clogging, and the absence of force or pressure sensing during puncture. These should be acknowledged directly in the poster; doing so makes the novelty claim more credible because it separates the implemented robotic infrastructure from future closed-loop autonomy.

## References

1. Wang W., Liu X., Gelinas D., Ciruna B., Sun Y. "A Fully Automated Robotic System for Microinjection of Zebrafish Embryos." *PLOS ONE*, 2007. https://doi.org/10.1371/journal.pone.0000862
2. Guo Z. et al. "Design of an automated robotic microinjection system for batch injection of zebrafish embryos and larvae." *Microsystems & Nanoengineering*, 2024. https://www.nature.com/articles/s41378-023-00645-6
3. Guo X. et al. "Design and developing a robot-assisted cell batch microinjection system for zebrafish embryo." *Microsystems & Nanoengineering*, 2025. https://www.nature.com/articles/s41378-024-00809-y
4. Zhao Y. et al. "A Review of Automated Microinjection of Zebrafish Embryos." *Micromachines*, 2019. https://doi.org/10.3390/mi10010007
