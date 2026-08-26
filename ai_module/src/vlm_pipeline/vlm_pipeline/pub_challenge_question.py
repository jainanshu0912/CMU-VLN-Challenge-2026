"""Publish /challenge_question once after a subscriber is visible, then exit.

Eval sends the same topic at 1 Hz; this is only for local one-shot tests.
``ros2 topic pub --once`` hangs here when FastDDS never reports a match.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

CHALLENGE_QUESTION_TOPIC = "/challenge_question"
QUESTION_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


class OncePublisher(Node):
  def __init__(self, text: str, wait_sec: float) -> None:
    super().__init__("challenge_question_once")
    self._text = text
    self._wait_sec = wait_sec
    self._pub = self.create_publisher(String, CHALLENGE_QUESTION_TOPIC, QUESTION_QOS)

  def publish_once(self) -> bool:
    deadline = time.monotonic() + self._wait_sec
    matched = False
    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)
      if int(self._pub.get_subscription_count()) > 0:
        matched = True
        break
    msg = String()
    msg.data = self._text
    self._pub.publish(msg)
    # Let FastDDS flush the sample before we tear down the participant.
    end = time.monotonic() + 0.5
    while time.monotonic() < end and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.05)
    return matched


def main(args: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Wait for a /challenge_question subscriber, publish once, exit."
  )
  parser.add_argument(
    "question",
    nargs="?",
    default="How many sofas are below a window?",
    help="Question text",
  )
  parser.add_argument(
    "--wait-sec",
    type=float,
    default=10.0,
    help="Seconds to wait for a subscriber before publishing anyway",
  )
  parsed, ros_args = parser.parse_known_args(args)

  text = parsed.question.strip()
  if not text:
    print("Empty question.", file=sys.stderr)
    sys.exit(1)

  rclpy.init(args=ros_args)
  node = OncePublisher(text, parsed.wait_sec)
  try:
    matched = node.publish_once()
    count = int(node._pub.get_subscription_count())
    extra = f"subscribers={count}" if matched else f"no match after {parsed.wait_sec:.0f}s, sent anyway"
    node.get_logger().info(
      f"Published once on {CHALLENGE_QUESTION_TOPIC} ({extra}): {text}"
    )
  finally:
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
