# Evaluator instructions

Use the official GPU flow in `docker/README.md`. The scored command is unchanged:

```bash
ros2 launch dummy_vlm dummy_vlm.launch
```

That launch **explores the scene, builds a live graph, then answers** `/challenge_question`. No VLA-3D CSV/JSON is required at test time.

## Extra steps for this entry

1. **GPU compose only.** CPU `compose.yml` cannot run GroundingDINO.

2. **Prefer the Docker Hub image** (avoids a multi-GB local build):

   [https://hub.docker.com/r/jainanshu0912/cmu-vln-ai-module](https://hub.docker.com/r/jainanshu0912/cmu-vln-ai-module)

   ```bash
   docker pull jainanshu0912/cmu-vln-ai-module:latest
   docker tag jainanshu0912/cmu-vln-ai-module:latest docker-ai_module:latest
   cd docker
   docker compose -f compose_gpu.yml up -d
   ```

   Do **not** pass `--build` after pulling. If you must build from this repo instead, use `docker compose -f compose_gpu.yml up --build -d` and leave **~25 GB** free.

3. **Start the system, then click Resume Navigation.**  
   In the system container run `system_simulation.sh`. In RViz, click **Resume Navigation to Goal** (Waypoint mode) so autonomy accepts `/way_point_with_heading`. Then start `dummy_vlm.launch` in `iros2026_ai_module`.

4. **Wait through exploration.**  
   The robot tours a few viewpoints before publishing a count, cube marker, or navigate path. No scored output during the tour is expected. The 1 Hz `/challenge_question` is buffered and handled **once** per launch.

5. **Do not start a second AI launch.**  
   Do not also run `vlm_pipeline.launch.py` or `pipeline_b.launch.py`. Scored topics are only `/way_point_with_heading`, `/selected_object_marker` (`CUBE`, `map`), and `/numerical_response`.
