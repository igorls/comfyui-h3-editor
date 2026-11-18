"""Run the cancellation helper at its real queue boundary without booting ComfyUI."""
import ast
import pathlib
import unittest
from unittest import mock

import project


def cancellation(comfy, queue):
    tree = ast.parse((pathlib.Path(__file__).parents[1] / "editor_api.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "stop_renders")
    namespace = {"comfy": comfy, "comfy_queue": queue, "store": project, "Refused": RuntimeError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "editor_api.py", "exec"), namespace)
    return namespace["stop_renders"]


class CancelRendersTest(unittest.IsolatedAsyncioTestCase):
    async def test_finals_leave_drafts_and_other_projects_alone(self):
        data = {"pending": [{"id": "f1", "quality": "final"}, {"id": "f2", "quality": "final"}, {"id": "d1", "quality": "draft"}]}
        comfy = mock.AsyncMock(return_value=(200, {}))
        stop = cancellation(comfy, lambda: ({"other-project"}, {"d1"}))
        self.assertEqual(await stop(data, "final"), ["f1", "f2"])
        comfy.assert_awaited_once_with("POST", "/queue", {"delete": ["f1", "f2"]})
        self.assertEqual([row["id"] for row in data["pending"]], ["d1"])

    async def test_queue_is_deleted_before_rechecking_the_running_prompt(self):
        data = {"pending": [{"id": "f1", "quality": "final"}, {"id": "f2", "quality": "final"}]}
        calls = []

        async def comfy(method, path, body):
            calls.append(path)
            return 200, {}

        def queue():
            self.assertEqual(calls, ["/queue"])
            return {"f2"}, set()

        await cancellation(comfy, queue)(data)
        self.assertEqual(calls, ["/queue", "/interrupt"])
        self.assertEqual(data["pending"], [])

    async def test_a_failed_delete_keeps_pending_jobs_available_for_retry(self):
        data = {"pending": [{"id": "f1", "quality": "final"}]}
        stop = cancellation(mock.AsyncMock(return_value=(500, {})), lambda: ({"f1"}, set()))
        with self.assertRaisesRegex(RuntimeError, "try again"):
            await stop(data)
        self.assertEqual(len(data["pending"]), 1)
