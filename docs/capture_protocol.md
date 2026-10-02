# Capture protocol (one page)

Route 2: stock apps. Pick ONE tier per capture. Before you start: switch on every light, open interior doors fully, and leave pets and people outside the rooms being scanned.

## A. LiDAR tier (iPhone or iPad **Pro** with LiDAR, iPhone 12 Pro or newer)
1. **Install** "Stray Scanner" (free, App Store, by Stray Robots). Open it once and allow camera access.
2. **Start** in the doorway of the first room. Tap the red record button.
3. **Hold** the phone upright at chest height, screen facing you. Walk slowly: about one step per second.
4. **In every room**, in this order:
   - walk once around the room about 1 m from the walls, camera pointed at the walls;
   - stop in the middle, tilt the phone up and sweep the **whole ceiling** once (ceiling height needs this);
   - tilt down and sweep where the walls meet the floor;
   - before leaving, stand 1-2 m in front of each doorway and look straight through it for 2 seconds.
5. **Go through every doorway slowly**, then do step 4 in the next room. Cover all rooms in **one** recording.
6. **Finish where you started**: walk back to the first doorway and point at the same wall you began with. Tap stop.
7. Expected length: 1-2 minutes per room.

## B. Video tier (any iPhone 15 or newer)
1. Settings > Camera > Formats > **Most Compatible**. Settings > Camera > Record Video > **1080p at 30 fps**.
2. Camera app, Video mode, phone held **sideways (landscape)** at chest height.
3. Walk the same route as in A (steps 2-6) in one clip, **slower than feels natural**, and turn slowly (about 5 seconds for a half turn).
   Keep furniture, door frames or corners in view: a frame showing only a blank wall cannot be placed.

## C. Photo tier (any iPhone 15 or newer)
1. Camera app, Photo mode, landscape, chest height, the normal 1x lens (not 0.5x). HEIC or JPEG both work.
2. **Per room, 6-8 photos**: from each corner, aim at the opposite corner; then one from the middle of each long wall facing the other wall.
   Consecutive photos must share about a third of their view.
3. **Doorways**: for every door, stand 1 m inside each room and photograph straight through the door, so the next room is visible. Put the photo in the folder of the room you stood in.
4. Make one folder per room on your computer, named `room1`, `room2`, ... and put all room folders inside one folder.

## Avoid (all tiers)
- **Mirrors and glass** (shower screens, glass doors, windows): do not point at them for long; they create fake openings and fake rooms.
- **Wet or glossy floors** and **low light**: dry and light the room first; dark scenes lose depth and features.
- Fast turns and walking backwards; covering the camera or LiDAR with a finger; people walking through the view.

## Hand the files over
- **LiDAR**: open Files > On My iPhone > Stray Scanner, find the newest folder (a code like `c7d28f72c6`), share it to the computer (AirDrop or cable). It must contain `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`.
- **Video**: AirDrop the clip (Options > "All Photos Data" on). **Photos**: the folder of room folders from C4.
- Then on the computer, from the repo folder: `uv run --group models plan run <folder or clip>`. Results appear in `out/<tier>_<name>/` as `plan.json` and `plan.png`.
