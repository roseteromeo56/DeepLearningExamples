import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"
TRAIN = "/workspace/moflow_pyt/moflow/runtime/train.py"
EVALUATE = "/workspace/moflow_pyt/moflow/runtime/evaluate.py"


class TrainScriptTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="moflow-train-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.log = self.root / "commands.jsonl"
        self.marker = self.root / "injected"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        stub = f"#!{sys.executable}\n" + '''import json
import os
import sys

with open(os.environ["MOFLOW_TEST_LOG"], "a") as output:
    output.write(json.dumps({"program": os.path.basename(sys.argv[0]),
                             "args": sys.argv[1:]}) + "\\n")
if os.path.basename(sys.argv[0]) == "torchrun" or sys.argv[1].endswith("train.py"):
    variable = "MOFLOW_TEST_TRAIN_EXIT"
else:
    variable = "MOFLOW_TEST_EVAL_EXIT"
sys.exit(int(os.environ.get(variable, "0")))
'''
        for name in ("python", "torchrun"):
            executable = bin_dir / name
            executable.write_text(stub)
            executable.chmod(0o755)
        self.env = dict(os.environ)
        self.env.update(
            PATH=str(bin_dir) + os.pathsep + os.defpath,
            MOFLOW_TEST_LOG=str(self.log),
            MOFLOW_TEST_TRAIN_EXIT="0",
            MOFLOW_TEST_EVAL_EXIT="0",
        )

    def run_script(self, *arguments, script=SCRIPT):
        result = subprocess.run(
            ["/bin/bash", str(script), *arguments],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        calls = []
        if self.log.exists():
            calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, calls

    def assert_commands(self, calls, gpus="1", precision="amp", flags=()):
        train = [TRAIN, "--cuda_graph", *flags]
        evaluate = [EVALUATE, "--steps", "1000", "--jit", *flags]
        if precision == "amp":
            train.append("--amp")
            evaluate.append("--amp")
        if gpus == "1":
            train.extend(["--learning_rate", "0.0001"])
        else:
            train.insert(0, f"--nproc_per_node={gpus}")
        self.assertEqual(calls, [
            {"program": "python" if gpus == "1" else "torchrun", "args": train},
            {"program": "python", "args": evaluate},
        ])

    def test_shell_syntax(self):
        result = subprocess.run(["/bin/bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_defaults(self):
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_commands(calls)

    def test_documented_configurations(self):
        for arguments, gpus, precision in [
            (("8",), "8", "amp"),
            (("4", "full"), "4", "full"),
            (("1", "full"), "1", "full"),
            (
                ("8", "amp", "--batch_size", "2048", "--log_path", "/results/dll.json"),
                "8", "amp",
            ),
        ]:
            with self.subTest(arguments=arguments):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_script(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_commands(calls, gpus, precision, arguments[2:])

    def test_extra_arguments_remain_literal(self):
        marker = shlex.quote(str(self.marker))
        flags = (
            "--log_path", "a path with spaces.json", "*.json", "",
            f"; touch {marker}", f"$(touch {marker})", f"`touch {marker}`",
            "single'quote", 'double"quote', "line\nbreak", "back\\slash",
        )
        (self.root / "glob.json").touch()
        result, calls = self.run_script("1", "amp", *flags)
        self.assertFalse(self.marker.exists(), "An argument was executed as shell code")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_commands(calls, flags=flags)

    def test_gpu_argument_remains_literal(self):
        gpus = f"2; touch {shlex.quote(str(self.marker))}"
        result, calls = self.run_script(gpus, "full")
        self.assertFalse(self.marker.exists(), "GPU count was executed as shell code")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_commands(calls, gpus, "full")

    def test_inherited_commands_cannot_execute(self):
        payload = f"touch {shlex.quote(str(self.marker))}"
        self.env.update(cmd=payload, eval_cmd=payload)
        result, calls = self.run_script("1", "full")
        self.assertFalse(self.marker.exists(), "An inherited command was executed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_commands(calls, "1", "full")

    def test_script_path_remains_literal(self):
        script = self.root / "train'$(touch injected)'.sh"
        script.write_bytes(SCRIPT.read_bytes())
        result, calls = self.run_script(script=script)
        self.assertFalse(self.marker.exists(), "Script path was executed as shell code")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_commands(calls)

    def test_training_failure_skips_evaluation(self):
        self.env["MOFLOW_TEST_TRAIN_EXIT"] = "37"
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"][0], TRAIN)

    def test_evaluation_failure_is_returned(self):
        self.env["MOFLOW_TEST_EVAL_EXIT"] = "23"
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assert_commands(calls)


if __name__ == "__main__":
    unittest.main()
