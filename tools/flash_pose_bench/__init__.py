"""Opt-in Flash Ref2VA pose adapter for reproducing the motion transfer experiments."""

import torch
from comfy_api.latest import io


class H3FlashPoseBench(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3FlashPoseBench",
            display_name="H3 Flash Pose Benchmark",
            category="minimax/h3",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.ModelPatch.Input("model_patch"),
                io.Image.Input("control_video"),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, vae, model_patch, control_video, strength):
        import folder_paths
        from comfy.patcher_extension import WrappersMP
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlBlockPatch, MiniMaxH3FunControlPatch
        from safetensors import safe_open

        backbone = model.get_model_object("diffusion_model")
        if len(backbone.blocks) != 49 or not hasattr(backbone, "endpoint_time_embedder"):
            raise ValueError("Expected HyperFlow Flash Ref2VA with 49 transformer blocks")
        if model_patch.model.injection_layers != (0, 10, 20, 30, 40):
            raise ValueError("Expected the five-block MiniMax H3 Fun ControlNet v1")
        if strength == 0:
            return io.NodeOutput(model)

        curve_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        with safe_open(curve_path, framework="pt", device="cpu") as checkpoint:
            curve = checkpoint.get_tensor("adaln_t_table")

        class FlashPosePatch(MiniMaxH3FunControlPatch):
            def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options=None, **kwargs):
                original = backbone._time_embedding
                previous_override = vars(backbone).get("_time_embedding")
                self.control_time = None

                def embed(unique_t, device, dtype):
                    # ControlNet needs the same timestep row order as Flash's mod_segments.
                    if self.active:
                        table = curve.to(device=device)
                        times = torch.tensor(unique_t, device=device, dtype=torch.float32)
                        position = times.clamp(0, 1) * (len(table) - 1)
                        lower = position.floor().long().clamp(max=len(table) - 2)
                        self.control_time = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
                    return original(unique_t, device, dtype)

                backbone._time_embedding = embed
                try:
                    return super().diffusion_model_wrapper(executor, x, timestep, context, transformer_options or {}, **kwargs)
                finally:
                    if previous_override is None:
                        del backbone._time_embedding
                    else:
                        backbone._time_embedding = previous_override
                    self.control_time = None

            def after_block(self, block_index, args, out):
                if self.active:
                    if self.control_time is None or self.control_time.shape != (args["t_emb"].shape[0], 8):
                        raise RuntimeError("Flash and ControlNet timestep rows did not align")
                    args = dict(args, t_emb=self.control_time)
                return super().after_block(block_index, args, out)

        result = model.clone()
        sampling = result.get_model_object("model_sampling")
        patch = FlashPosePatch(model_patch, vae, control_video[..., :3].movedim(-1, 1), None, None,
                               strength, float(sampling.percent_to_sigma(0)), float(sampling.percent_to_sigma(1)))
        result.add_wrapper(WrappersMP.DIFFUSION_MODEL, patch.diffusion_model_wrapper)
        blocks = result.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for original_index, flash_index in zip(model_patch.model.injection_layers, (0, 9, 19, 29, 39), strict=True):
            replaced = MiniMaxH3FunControlBlockPatch(patch, original_index, blocks.get(("double_block", flash_index)))
            result.set_model_patch_replace(replaced, "dit", "double_block", flash_index)
        return io.NodeOutput(result)


NODE_CLASS_MAPPINGS = {"H3FlashPoseBench": H3FlashPoseBench}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FlashPoseBench": "H3 Flash Pose Benchmark"}
