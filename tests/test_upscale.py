import unittest

import project as store
from project import add_clip, add_pending, finish_render, new_project, plan_render

FILES = {"relayvsr/relayvsr_x2_lightx2v_int8_convrot.safetensors", "relayvsr/relayvsr_flash_vae_x2_fp32.safetensors"}


def rendered(quality="final", frames=124):
    """A project with one clip that has one finished take."""
    project = new_project("Test")
    clip, _ = add_clip(project, {"action": "One."})
    job, _ = plan_render(project, "Test", clip["id"], quality)
    job["id"] = "prompt-1"
    add_pending(project, job)
    finish_render(project, "prompt-1", {"filename": "x.mp4", "subfolder": "h3take_projects/Test/clips"}, {"frames": frames, "poster": "x.jpg"})
    return project, clip


class UpscaleTest(unittest.TestCase):
    def test_an_upscale_reads_the_final_and_saves_a_take_at_twice_the_size(self):
        project, clip = rendered()
        job, graph = plan_render(project, "Test", clip["id"], "upscale", available=FILES)
        self.assertEqual(graph["30"]["inputs"]["file"], "h3take_projects/Test/clips/x.mp4")
        self.assertEqual(graph["31"]["inputs"]["unet_name"], "relayvsr/relayvsr_x2_lightx2v_int8_convrot.safetensors")
        self.assertEqual(graph["32"]["inputs"]["vae_name"], "relayvsr/relayvsr_flash_vae_x2_fp32.safetensors")
        self.assertEqual(graph["19"]["class_type"], "SaveVideo")
        self.assertIn("%s_%s" % (clip["id"], job["take"]), graph["19"]["inputs"]["filename_prefix"])
        self.assertEqual(job["size"], [3840, 2176])
        self.assertEqual(job["loads"], "relayvsr/relayvsr_x2_lightx2v_int8_convrot.safetensors")

    def test_the_upscaled_take_is_a_final_that_names_where_it_came_from(self):
        project, clip = rendered()
        source = store.active_take(clip)
        job, _ = plan_render(project, "Test", clip["id"], "upscale", available=FILES)
        job["id"] = "prompt-2"
        add_pending(project, job)
        finish_render(project, "prompt-2", {"filename": "y.mp4", "subfolder": "h3take_projects/Test/clips"}, {"frames": 124, "poster": "y.jpg"})
        take = store.active_take(clip)
        self.assertEqual((take["quality"], take["size"], take["upscaled"]), ("final", [3840, 2176], source["id"]))
        self.assertEqual(store.take_size(project, take), (3840, 2176))
        self.assertEqual(take["models"]["profile"], source["models"]["profile"])
        self.assertEqual(len(clip["takes"]), 2)

    def test_an_upscaled_take_is_not_upscaled_again(self):
        project, clip = rendered()
        job, _ = plan_render(project, "Test", clip["id"], "upscale", available=FILES)
        job["id"] = "prompt-2"
        add_pending(project, job)
        finish_render(project, "prompt-2", {"filename": "y.mp4", "subfolder": "s"}, {"frames": 124, "poster": "y.jpg"})
        with self.assertRaisesRegex(ValueError, "already upscaled"):
            plan_render(project, "Test", clip["id"], "upscale", available=FILES)

    def test_a_draft_is_not_upscaled(self):
        project, clip = rendered("draft")
        with self.assertRaisesRegex(ValueError, "Render a final first"):
            plan_render(project, "Test", clip["id"], "upscale", available=FILES)

    def test_a_clip_with_no_render_is_refused(self):
        project = new_project("Test")
        clip, _ = add_clip(project, {"action": "One."})
        with self.assertRaisesRegex(ValueError, "no render"):
            plan_render(project, "Test", clip["id"], "upscale", available=FILES)

    def test_a_long_clip_is_refused(self):
        project, clip = rendered(frames=300)
        with self.assertRaisesRegex(ValueError, "300 frames"):
            plan_render(project, "Test", clip["id"], "upscale", available=FILES)

    def test_missing_files_are_named(self):
        project, clip = rendered()
        with self.assertRaisesRegex(ValueError, "relayvsr_flash_vae_x2_fp32.safetensors"):
            plan_render(project, "Test", clip["id"], "upscale", available={"relayvsr_x2_lightx2v_int8_convrot.safetensors"})

    def test_files_in_the_models_root_are_found_too(self):
        project, clip = rendered()
        flat = {"relayvsr_x2_lightx2v_int8_convrot.safetensors", "relayvsr_flash_vae_x2_fp32.safetensors"}
        _, graph = plan_render(project, "Test", clip["id"], "upscale", available=flat)
        self.assertEqual(graph["31"]["inputs"]["unet_name"], "relayvsr_x2_lightx2v_int8_convrot.safetensors")


if __name__ == "__main__":
    unittest.main()
