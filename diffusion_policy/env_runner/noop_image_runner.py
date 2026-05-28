from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NoopImageRunner(BaseImageRunner):
    """No-op runner for datasets that cannot be evaluated in a simulator (e.g. real demos)."""

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
