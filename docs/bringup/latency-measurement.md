# Camera capture latency: measure it once before calling any latency end-to-end

WO-07's benchmark is latency from the optical event to the first bus write, target < 150 ms.
The deploy client logs **`frame_return_to_first_write_ms`**, which is **not** that metric. It is
never reported as end-to-end until the capture offset below has been measured and written to
`configs/real.yaml` `deploy.capture_offset_ms`.

## Why the logged metric excludes capture (checked in code; no camera was opened)

- LeRobot 0.4.4 stamps a frame in its background read thread with `time.perf_counter()` after
  `VideoCapture.read()` has returned and after the BGR->RGB post-processing
  (`.venv/lib/python3.11/site-packages/lerobot/cameras/opencv/camera_opencv.py:443-451`,
  `_read_loop`). `src/robot/camera.py` passes that stamp on as `Frame.capture_time`. So the stamp
  marks frame **return** on the host clock. It does not mark capture.
- The installed OpenCV is 4.11.0 (`opencv-python-headless`). Its AVFoundation backend on macOS
  does not expose the sample buffer's presentation timestamp. `CAP_PROP_POS_MSEC` is not
  meaningful for a live camera. Agent finding from OpenCV's source and documentation; not
  measured on this Mac.
- What the stamp excludes: exposure, sensor readout, USB transfer, AVFoundation / driver
  buffering and decode. On a USB webcam these terms can total tens of ms, and more if
  auto-exposure is on. **No exposure lock exists yet**: exposure / white-balance / focus locking
  is not implemented (project review gap 4, open, out of scope for PR #29), and no WO-06
  checklist step covers it. Until it is, keep `deploy.capture_offset_ms: null`, so the client
  keeps printing `end-to-end latency: NOT measured` and no `end_to_end_estimate_ms` is reported.
  The capture timestamp itself is not verified either: the procedure below measures an offset,
  it does not make the driver stamp capture.

A driver-side capture timestamp aligned with the host clock is therefore not available without
new code below LeRobot. Measure the offset optically once instead.

## Procedure: on-screen clock filmed by the robot camera (~10 min, no arm motion)

`time.perf_counter()` on macOS reads a system-wide monotonic clock (`mach_absolute_time`), so
two Python processes on the same Mac share it. One process shows the clock on screen. The
other saves camera frames named by their LeRobot stamp. The offset for each frame is
`stamp - clock value visible in the frame`.

1. Prerequisites: WO-06 step 6 (camera index, `cameras.front` in `configs/real.yaml`), and an
   exposure lock that will also be in force for WO-07. That lock does not exist yet (gap 4,
   above): do this measurement only once it does. A value measured under auto-exposure is
   invalid by the rule below; leave `deploy.capture_offset_ms` null until then. The arm can stay
   unpowered.
2. Terminal A: a large running clock. Zoom the terminal font (Cmd +) until the digits fill
   about a third of the camera view, then point the robot camera at the screen:

   ```bash
   uv run python -c "import sys, time
   while True:
       sys.stdout.write(f'\r{time.perf_counter() * 1e3:12.1f}'); sys.stdout.flush(); time.sleep(0.002)"
   ```

3. Terminal B: save 40 frames, each named by its LeRobot stamp in ms:

   ```bash
   source scripts/mac_env.sh
   uv run python -c "
   import pathlib, cv2
   from src.config import load_config
   from src.robot.camera import OpenCVFrameSource
   out = pathlib.Path('runs/latency-offset'); out.mkdir(parents=True, exist_ok=True)
   with OpenCVFrameSource(load_config('real')['cameras']['front']) as cam:
       for _ in range(40):
           f = cam.read()
           cv2.imwrite(str(out / f'{f.capture_time * 1e3:.1f}.png'), cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR))
   "
   ```

4. Open about 20 of the PNGs. For each one, `offset = filename - clock digits in the image` (ms).
   Skip frames where the digits are blurred between two values.
5. Write the **median** offset to `configs/real.yaml` `deploy.capture_offset_ms`. Record the median,
   the 90th percentile, the camera, the exposure setting and the date in `docs/STATUS.md`.

What the number means. The clock in the image was computed before the screen showed it. Screen
refresh and terminal rendering add roughly 10-30 ms, so the measured offset is an **upper bound**
on capture latency, which is the conservative direction for a "< 150 ms" target. It is valid only
for the exposure, resolution and fps it was measured with. Re-measure when any of them changes.

## What the client reports afterwards

With `deploy.capture_offset_ms` set, `src/robot/client.py` `summarize` adds
`end_to_end_estimate_ms` = `frame_return_to_first_write_ms` + the offset. The estimate carries the
offset and the method in the record. With the offset `null`, it reports
`end_to_end_estimate_ms: null` and prints `end-to-end latency: NOT measured`.
`frame_return_p95_under_target` is necessary but not sufficient for the WO-07 target.

Optional cross-check (order of magnitude only): during a WO-07 pilot trial, film the screen clock
and the gripper together in one 240 fps phone video. Count the frames from a change in the scene
to the arm's first visible reaction. That total also includes inference and servo response.

Status: **unverified on hardware.** No camera has been opened for this measurement.
