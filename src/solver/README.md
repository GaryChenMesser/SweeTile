Problem one:
Track all the required tiles with desired resolution (4k or 2k).
  1. Need a correct projection to make sure all the required edge tiles are included. --> Write a new rotation function from scratch.
  2. Able to change field of view in response to near-periphery / mid-periphery / far-periphery --> Add option to the rotation function.
  3. Draw it out to verify --> Debug using matplotlib3D.

Problem two:
Retrieve actual payload size of each tile (tiles with low and high latitude should vary in size?)
  1. Use GPAC to discard different tiles to make sure the actual size of each tile.
