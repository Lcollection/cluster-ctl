import unittest

from cluster_manager.cli import ssh_command


class SSHCommandTests(unittest.TestCase):
    def setUp(self):
        self.node = {
            "host": "gpu.example.org",
            "port": 22,
            "user": "researcher",
            "work_root": "~/ai-training",
            "project_dir": "projects/model one",
        }

    def test_quotes_project_path_and_arguments(self):
        command = ssh_command(self.node, ["train", "python", "train.py", "--label", "a b"])
        self.assertEqual(command[:5], ["ssh", "-p", "22", "-o", "BatchMode=yes"])
        self.assertEqual(command[5], "researcher@gpu.example.org")
        self.assertIn("export PART_ROOT=\"$HOME\"/ai-training", command[6])
        self.assertIn("\"$PART_ROOT\"/'projects/model one'", command[6])
        self.assertIn("'a b'", command[6])

    def test_shell_forces_a_terminal(self):
        command = ssh_command(self.node, ["shell"], interactive=True)
        self.assertIn("-tt", command)

    def test_identity_file_is_expanded(self):
        self.node["identity_file"] = "~/.ssh/id_test"
        command = ssh_command(self.node, ["status"])
        self.assertIn("-i", command)
        self.assertTrue(command[command.index("-i") + 1].endswith("/.ssh/id_test"))


if __name__ == "__main__":
    unittest.main()
