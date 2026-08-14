import os

from setuptools import find_packages, setup

package_name = "vlm_pipeline_sort3d"
share_dir = os.path.join("share", package_name)

setup(
  name=package_name,
  version="0.0.1",
  packages=find_packages(),
  data_files=[
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    (share_dir, ["package.xml"]),
    (os.path.join(share_dir, "launch"), [
      "launch/pipeline_c.launch.py",
    ]),
    (os.path.join(share_dir, "config"), [
      "config/pipeline_c.yaml",
    ]),
  ],
  install_requires=["setuptools", "numpy"],
  zip_safe=True,
  maintainer="Anshu Jain",
  maintainer_email="anshuj@andrew.cmu.edu",
  description="Pipeline C SORT3D-style reasoning for CMU VLN Challenge",
  license="BSD",
  tests_require=["pytest"],
  entry_points={
    "console_scripts": [
      "sort3d_node = vlm_pipeline_sort3d.main_node:main",
    ],
  },
)
