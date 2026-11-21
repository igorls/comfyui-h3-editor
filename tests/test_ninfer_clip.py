"""Tests for ninfer_clip (remote NInfer text/vision encoder wire). Run inside the container."""

import http.server
import json
import os
import struct
import tempfile
import threading
import unittest

try:
    import torch
    from safetensors.torch import save_file

    import ninfer_clip
except ImportError:
    torch = None


def _pack_response(meta, tensor_bf16):
    header = json.dumps(meta).encode("utf-8")
    raw_bytes = tensor_bf16.contiguous().view(torch.uint16).cpu().numpy().tobytes()
    return struct.pack("<Q", len(header)) + header + raw_bytes


@unittest.skipIf(torch is None, "needs torch")
class NInferClipTest(unittest.TestCase):
    def test_decode_hidden_response_and_token_tags(self):
        torch.manual_seed(42)
        h = torch.randn(4, 16, dtype=torch.bfloat16)
        meta = {
            "model": "qwen3.8-27b",
            "seq_len": 4,
            "hidden_size": 16,
            "dtype": "bf16",
            "layer": 48,
            "token_types": [0, 1, 1, 0],
        }
        parsed_meta, tensor = ninfer_clip.decode_hidden_response(_pack_response(meta, h))
        self.assertEqual(parsed_meta["layer"], 48)
        self.assertEqual(tuple(tensor.shape), (1, 4, 16))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.allclose(tensor[0], h.float(), atol=1e-4))
        tags = ninfer_clip.token_tags_from_types(4, parsed_meta["token_types"])
        self.assertEqual(tags.tolist(), [0, 0, 0, 0])
        toks = ["a", "<|vision_start|>", "<|image_pad|>", "<|image_pad|>", "<|vision_end|>", "b"]
        tags2 = ninfer_clip.token_tags_from_types(6, [0, 0, 1, 1, 0, 0], toks)
        self.assertEqual(tags2.tolist(), [1, 0, 0, 0, 0, 1])

    def test_dual_modality_projection_routes_vision_and_text_separately(self):
        d = 8
        proj = {
            "tap": 51,
            "W": torch.eye(d),
            "mean_in": torch.zeros(d),
            "std_in": torch.ones(d),
            "mean_out": torch.zeros(d),
            "std_out": torch.ones(d),
            "sink_out": torch.full((d,), 9.0),
            "W_vis": torch.eye(d) * 2.0,
            "mean_in_vis": torch.zeros(d),
            "std_in_vis": torch.ones(d),
            "mean_out_vis": torch.zeros(d),
            "std_out_vis": torch.ones(d),
        }
        h = torch.ones(1, 4, d)
        tags = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        out = ninfer_clip.project_with_modality(h, proj, tags, {})
        self.assertTrue(torch.allclose(out[0, 0], torch.full((d,), 9.0)))
        self.assertTrue(torch.allclose(out[0, 1], torch.full((d,), 2.0)))
        self.assertTrue(torch.allclose(out[0, 2], torch.full((d,), 2.0)))
        self.assertTrue(torch.allclose(out[0, 3], torch.full((d,), 1.0)))

    def test_find_user_span_strips_qwen_chat_wrapper(self):
        toks = ["<|im_start|>", "user", "\n", "red", " car", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
        self.assertEqual(ninfer_clip.find_user_span(toks, len(toks)), (3, 5))
        merged = ["<|im_start|>", "user\n", "red", " car", "<|im_end|>"]
        self.assertEqual(ninfer_clip.find_user_span(merged, len(merged)), (2, 4))
        raw = ["red", " car"]
        self.assertEqual(ninfer_clip.find_user_span(raw, len(raw)), (0, 2))

    def test_build_parts_text_and_reference_images(self):
        img = torch.zeros(1, 16, 16, 3)
        parts = ninfer_clip.build_parts(
            "a cat on a rooftop",
            minimax_ref_items=[{"type": "image", "data": img}, {"type": "audio"}],
        )
        self.assertEqual([p["type"] for p in parts], ["text", "image_url", "text"])
        self.assertEqual(parts[0]["text"], "<Picture 1>: ")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(parts[2]["text"], "<Audio 1>: a cat on a rooftop")

    def test_remote_clip_roundtrip_and_optional_projection(self):
        seen = {}
        expected = torch.arange(3 * 8, dtype=torch.float32).view(3, 8).to(torch.bfloat16)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                seen["path"] = self.path
                seen["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
                meta = {
                    "model": "qwen3.8-27b",
                    "seq_len": 3,
                    "hidden_size": 8,
                    "dtype": "bf16",
                    "layer": seen["body"].get("layer"),
                    "token_types": [0, 0, 0],
                    "prefill_ms": 1.5,
                }
                body = _pack_response(meta, expected)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = "http://127.0.0.1:%d" % port
            clip = ninfer_clip.NInferRemoteCLIP(url=url, model="qwen3.8-27b", layer=48)
            tokens = clip.tokenize("hello world")
            scheduled = clip.encode_from_tokens_scheduled(tokens)
            cond, extra = scheduled[0]
            self.assertEqual(seen["path"], "/v1/hidden_states")
            self.assertEqual(seen["body"]["layer"], 48)
            self.assertEqual(tuple(cond.shape), (1, 3, 8))
            self.assertTrue(torch.allclose(cond[0].cpu(), expected.float(), atol=1e-4))
            self.assertEqual(extra["minimax_token_tags"].tolist(), [1, 1, 1])

            # With an identity projection file, layer falls back to proj['tap'] when layer=-1.
            folder = tempfile.mkdtemp()
            proj_path = os.path.join(folder, "id.safetensors")
            save_file(
                {
                    "W": torch.eye(8),
                    "mean_in": torch.zeros(8),
                    "std_in": torch.ones(8),
                    "mean_out": torch.zeros(8),
                    "std_out": torch.ones(8),
                },
                proj_path,
                metadata={"tap": "40"},
            )
            pclip = ninfer_clip.NInferRemoteCLIP(url=url, layer=-1, projection_path=proj_path)
            pcond = pclip.encode("projected")
            self.assertEqual(seen["body"]["layer"], 40)
            self.assertTrue(torch.allclose(pcond[0].cpu(), expected.float(), atol=1e-4))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
