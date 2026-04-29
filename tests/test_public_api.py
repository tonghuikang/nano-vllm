import os
import subprocess
import sys
import unittest


class PublicApiImportTest(unittest.TestCase):
    def test_sampling_params_import_does_not_validate_backend_env(self):
        env = os.environ.copy()
        env["NANO_VLLM_CASCADE_SUFFIX_KERNEL"] = "unknown"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from nanovllm import SamplingParams; print(SamplingParams.__name__)",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "SamplingParams")

    def test_llm_remains_public_export(self):
        from nanovllm import LLM

        self.assertEqual(LLM.__name__, "LLM")


if __name__ == "__main__":
    unittest.main()
