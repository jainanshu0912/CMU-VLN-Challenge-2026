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
      "launch/pipeline_b_cpu.launch.py",
      "launch/pipeline_b_manual.launch.py",
      "launch/pipeline_b_manual_cpu.launch.py",
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
    ],
  },
)
