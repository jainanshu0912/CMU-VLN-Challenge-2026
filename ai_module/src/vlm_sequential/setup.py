import os

from setuptools import find_packages, setup

package_name = "vlm_sequential"
share_dir = os.path.join("share", package_name)

setup(
  name=package_name,
  version="0.0.1",
  packages=find_packages(),
  data_files=[
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    (share_dir, ["package.xml"]),
    (
      os.path.join(share_dir, "launch"),
      ["launch/vlm_sequential.launch.py"],
    ),
  ],
  install_requires=["setuptools"],
  zip_safe=True,
  maintainer="Anshu Jain",
  maintainer_email="anshuj@andrew.cmu.edu",
  description="Sequential VLN pipeline: explore, build a scene graph, then answer questions",
  license="BSD",
  tests_require=["pytest"],
  entry_points={
    "console_scripts": [
      "vlm_sequential_node = vlm_sequential.sequential_node:main",
    ],
  },
)
