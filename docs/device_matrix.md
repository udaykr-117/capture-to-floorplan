# Device matrix

What each tier needs, where it runs, and what it honestly delivers. "Measured" numbers come from the three supplied LiDAR captures of one
property (no ground truth exists: all numbers are agreement between repeat captures or against the phone's own ARKit trajectory, not accuracy against
a tape). The capture device of the samples is unknown (1920x1440 RGB, 256x192 depth). No iPhone was available to us, so no tier
has been run on a capture we made ourselves.

| Tier | Phone | Capture app | Processing (our side) | What it delivers on the samples | Status |
|---|---|---|---|---|---|
| LiDAR | iPhone 12 Pro or newer Pro/Pro Max, iPad Pro with LiDAR (ARKit depth + poses) | Stray Scanner (free; the app the protocol names) | any laptop, CPU only; measured 9-174 s per capture depending on its length, + ~1-1.6 s per keyframe for the damage pass | Full plan: rooms, walls, areas, ceilings (where the ceiling was scanned), openings, adjacency, damage pass. Between repeat captures: room widths 3 of 12 within 1 cm / 0.5% (median 7.7 cm), same wall's length median 24.7 cm, room area median 1.04 m2 (15%), two whole-house footprints 4% apart. Wall-plane position error sigma ~3.5 cm (cross-capture). Openings found 3 / 7 / 1 in the three captures. Opening widths and ceiling heights: **untested** against ground truth | runs; accuracy gates not met or untested |
| Video | any iPhone 15 or newer (no LiDAR needed) | Camera app, 1080p30, Most Compatible | CPU; measured: SfM 327 s for 650 frames + depth model ~1.4 s per placed frame | **No plan on the samples**: structure from motion placed 4-18% of frames (low-texture walls, motion blur). Where frames are placed, the camera path matches ARKit to 2-3 cm; metric scale is uncertain by +-20% (1 sigma) and was off by -11%, -22% and +56% on the three samples. Optional learned matcher (DISK + LightGlue): 33% of single_room frames placed instead of 18%, still no plan, 36 min for 191 frames | runs, produces "no plan" with the reason |
| Photo | any iPhone 15 or newer | Camera app (HEIC or JPEG) | CPU; measured: SfM 13 s for 31 photos | **No plan on simulated photo sets** (8 and 31 stills chosen from the sample videos; 0 registered). Real photo sets untested | runs, produces "no plan" with the reason |

Expected behaviour on an unseen home (estimated, not measured):
- LiDAR: walls within a few cm (the cross-capture numbers above), worse in rooms the path crosses only once at the end of a long walk (drift), and
  in rooms with mirrors or glass (phantom openings). Ceiling heights only for rooms whose ceiling was swept (protocol step A4).
- Video and photo: a plan only for well-textured, well-lit homes with slow, overlapping capture; otherwise an explicit "no plan" result, never invented rooms.
