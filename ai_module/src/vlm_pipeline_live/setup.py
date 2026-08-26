import os

from setuptools import find_packages, setup

package_name = "vlm_pipeline_live"
share_dir = os.path.join("share", package_name)

setup(
  name=package_name,
  version="0.0.1",
  packages=find_packages(),
  data_files=[
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    (share_dir, ["package.xml"]),
    (os.path.join(share_dir, "launch"), [
      "launch/explorer.launch.py",
      "launch/live_detector.launch.py",
      "launch/live_scene_graph.launch.py",
      "launch/pipeline_b.launch.py",
      "launch/pipeline_b_manual.launch.py",
    ]),
    (os.path.join(share_dir, "data", "live_scene"), [
      "data/live_scene/README.md",
      "data/live_scene/live_scene_scene_graph.json",
      "data/live_scene/live_scene_object_result.csv",
      "data/live_scene/object_list.txt",
    ]),
    (os.path.join(share_dir, "data", "captured"), [
      "data/captured/README.md",
    ]),
  ],
  install_requires=["setuptools", "numpy"],
  zip_safe=True,
  maintainer="Anshu Jain",
  maintainer_email="anshuj@andrew.cmu.edu",
  description="Pipeline B live exploration for CMU VLN Challenge",
  license="BSD",
  tests_require=["pytest"],
  entry_points={
    "console_scripts": [
      "explorer_node = vlm_pipeline_live.explorer:main",
      "live_detector_node = vlm_pipeline_live.live_detector:main",
      "live_scene_graph_node = vlm_pipeline_live.live_scene_graph_node:main",
      "write_object_list_from_scene_graph = vlm_pipeline_live.write_object_list_from_scene_graph:main",
      "archive_captured_scene = vlm_pipeline_live.archive_captured_scene:main",
      "compare_scene_graphs = vlm_pipeline_live.compare_scene_graphs:main",
      "compare_backend_runs = vlm_pipeline_live.compare_backend_runs:main",
    ],
  },
)
