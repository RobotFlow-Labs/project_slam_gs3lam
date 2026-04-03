# PRD-06: ROS2 Integration

> Module: SLAM-GS3LAM | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
GS3LAM runs as a ROS2 component that subscribes to RGB-D and semantic streams and publishes poses, pointwise semantic field snapshots, and diagnostics for ANIMA robotics stacks.

## Context (from paper)
The paper assumes RGB-D plus semantics as input. ANIMA requires the same multimodal interface to be available on robotics middleware, with semantics supplied by an upstream perception node when the dataset does not provide labels.
**Paper reference**: §3.1 "process RGB-D data with ... 2D semantic labels"; §4.1.2 datasets

## Acceptance Criteria
- [ ] ROS2 node subscribes to image, depth, intrinsics, and semantic topics
- [ ] Node publishes pose estimates, rendered semantic map, and diagnostics
- [ ] Launch file supports replay from bag files and live RGB-D topics
- [ ] Test: `uv run pytest tests/test_ros2_contracts.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_gs3lam/ros2/node.py` | ROS2 runtime wrapper | §3.1 | ~200 |
| `src/anima_slam_gs3lam/ros2/messages.py` | topic contract helpers | — | ~90 |
| `configs/ros2.toml` | topic names and QoS | adaptation | ~70 |
| `launch/gs3lam.launch.py` | launch entrypoint | adaptation | ~80 |
| `tests/test_ros2_contracts.py` | contract tests | — | ~90 |

## Architecture Detail (from paper)
### Inputs
- `/camera/rgb/image`
- `/camera/depth/image`
- `/camera/info`
- `/perception/semantic_labels`

### Outputs
- `/slam/pose`
- `/slam/semantic_map`
- `/slam/diagnostics`

### Algorithm
```python
class GS3LAMNode(Node):
    def on_frame(self, rgb, depth, semantic, camera_info):
        result = self.service.step(...)
        self.publish_pose(result.pose)
```

## Dependencies
```toml
rclpy = "*"
sensor-msgs-py = "*"
cv-bridge = "*"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| ROS2 bag with RGB-D-semantics | bag | `/mnt/forge-data/datasets/slam/gs3lam/rosbags/` | create from benchmark replay |

## Test Plan
```bash
uv run pytest tests/test_ros2_contracts.py -v
ros2 launch launch/gs3lam.launch.py
```

## References
- Paper: §3.1, §4.1.2
- Depends on: PRD-05
- Feeds into: PRD-07
