import os

from setuptools import find_packages, setup

package_name = "vlm_pipeline"
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
          ["launch/vlm_pipeline.launch.py", "launch/publish_questions.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anshu Jain",
    maintainer_email="anshuj@andrew.cmu.edu",
    description="Zero-shot VLM pipeline for CMU VLN Challenge",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vlm_pipeline_node = vlm_pipeline.main_node:main",
            "publish_questions = vlm_pipeline.publish_questions:main",
        ],
    },
)
