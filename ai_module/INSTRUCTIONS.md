# Evaluator instructions

Use the official GPU flow in `docker/README.md`. The scored command is unchanged:

```bash
ros2 launch dummy_vlm dummy_vlm.launch
```

That launch **explores the scene, builds a live graph, then answers** `/challenge_question`. No VLA-3D CSV/JSON is required at test time.

## Extra steps for this entry

1. **GPU compose only.**  
   `docker compose -f compose_gpu.yml up --build -d`  
   CPU `compose.yml` cannot run GroundingDINO.

2. **First image build is large.**  
   The AI Dockerfile installs PyTorch (CUDA 12.4), GroundingDINO, and BERT. Leave **~25 GB** free and let `--build` finish. After that, no extra colcon step is needed.

3. **Start the system, then click Resume Navigation.**  
   In the system container run `system_simulation.sh`. In RViz, click **Resume Navigation to Goal** (Waypoint mode) so autonomy accepts `/way_point_with_heading`. Then start `dummy_vlm.launch` in `iros2026_ai_module`.

4. **Wait through exploration.**  
   The robot tours a few viewpoints before publishing a count, cube marker, or navigate path. No scored output during the tour is expected. The 1 Hz `/challenge_question` is buffered and handled **once** per launch.

5. **Do not start a second AI launch.**  
   Do not also run `vlm_pipeline.launch.py` or `pipeline_b.launch.py`. Scored topics are only `/way_point_with_heading`, `/selected_object_marker` (`CUBE`, `map`), and `/numerical_response`.
