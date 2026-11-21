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

    import clipproj
    import ninfer_clip
    import seq_connector
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

    def test_find_qi21_span_drops_system_and_trailing_think(self):
        toks = [
            "<|im_start|>", "system", "\n", "Comprehend", "<|im_end|>", "\n",
            "<|im_start|>", "user", "\n", "a", " cat", "<|im_end|>", "\n",
            "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n",
        ]
        self.assertEqual(ninfer_clip.find_qi21_span(toks, len(toks)), (6, 16))
        self.assertEqual(toks[6:16], ["<|im_start|>", "user", "\n", "a", " cat", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"])

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

    def test_decode_multilayer_hidden_response(self):
        torch.manual_seed(7)
        h = torch.randn(5, 3, 8, dtype=torch.bfloat16)
        meta = {
            "model": "qwen3.8-27b",
            "seq_len": 5,
            "num_layers": 3,
            "hidden_size": 8,
            "dtype": "bf16",
            "layers": [-2, 15, 63],
            "token_types": [0, 0, 0, 0, 0],
        }
        parsed, tensor = ninfer_clip.decode_hidden_response(_pack_response(meta, h))
        self.assertEqual(parsed["layers"], [-2, 15, 63])
        self.assertEqual(tuple(tensor.shape), (1, 5, 24))
        self.assertTrue(torch.allclose(tensor[0], h.view(5, 24).float(), atol=1e-4))

    def test_seq_connector_padding_invariance_and_safetensors_roundtrip(self):
        torch.manual_seed(11)
        d_in, d_out = 24, 16
        net = seq_connector.BidirectionalSeqConnector(
            d_in=d_in, d_out=d_out, d_model=32, n_layers=2, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0
        )
        net.eval()
        x = torch.randn(1, 5, d_in)
        # Zero-init output projection starts identically at 0
        self.assertTrue(torch.allclose(net(x), torch.zeros(1, 5, d_out)))
        # Give out_proj random weights and test padding mask invariance
        torch.nn.init.normal_(net.out_proj.weight, std=0.1)
        torch.nn.init.normal_(net.out_proj.bias, std=0.1)
        y_single = net(x)
        x_pad = torch.zeros(2, 9, d_in)
        x_pad[0, :5] = x[0]
        x_pad[0, 5:] = 999.0  # garbage in padded slots must not leak
        x_pad[1, :3] = torch.randn(3, d_in)
        mask = torch.zeros(2, 9, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :3] = True
        y_pad = net(x_pad, valid_mask=mask)
        self.assertTrue(torch.allclose(y_pad[0, :5], y_single[0], atol=1e-5))
        self.assertTrue(torch.allclose(y_pad[0, 5:], torch.zeros(4, d_out), atol=1e-6))

        # Roundtrip through clipproj safetensors save/load/project
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "seq_test.safetensors")
        tensors = {
            "W": torch.zeros(d_in, d_out, dtype=torch.float16),
            "mean_in": torch.zeros(d_in, dtype=torch.float16),
            "std_in": torch.ones(d_in, dtype=torch.float16),
            "mean_out": torch.zeros(d_out, dtype=torch.float16),
            "std_out": torch.ones(d_out, dtype=torch.float16),
        }
        tensors.update(seq_connector.state_dict_prefixed(net, dtype=torch.float32))
        meta = {
            "tap": "63",
            "layers": json.dumps([-2, 15, 63]),
            "seq_config": json.dumps(net.config),
            "target_profile": "qwen_image_21",
        }
        save_file(tensors, path, metadata=meta)
        proj = clipproj.load_projection(path)
        out = clipproj.project(x, proj, {})
        self.assertTrue(torch.allclose(out, y_single, atol=1e-4))

    def test_unpack_by_char_spans_handles_1toN_and_Nto1(self):
        # 1:N unpacking ('ARIA' -> ['AR', 'IA']) and N:1 collapsing (['<d', '>'] -> ['<d>'])
        src_texts = ["<d", ">", "PAD", "ARIA"]
        tgt_texts = ["<d>", "PAD", "AR", "IA"]
        src_h = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [10.0, 20.0], [50.0, 60.0]], dtype=torch.float32
        )
        out = ninfer_clip.unpack_by_char_spans(src_h, src_texts, tgt_texts)
        self.assertEqual(tuple(out.shape), (4, 2))
        self.assertTrue(torch.allclose(out[0], torch.tensor([2.0, 0.0])))     # mean of '<d' and '>'
        self.assertTrue(torch.allclose(out[1], torch.tensor([10.0, 20.0])))   # exact 'PAD'
        self.assertTrue(torch.allclose(out[2], torch.tensor([50.0, 60.0])))   # unpacked 'AR'
        self.assertTrue(torch.allclose(out[3], torch.tensor([50.0, 60.0])))   # unpacked 'IA'

    def test_seq_connector_with_sink_excludes_token_zero(self):
        torch.manual_seed(19)
        d = 16
        net = seq_connector.BidirectionalSeqConnector(
            d_in=d, d_out=d, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0
        )
        torch.nn.init.normal_(net.out_proj.weight, std=0.1)
        torch.nn.init.normal_(net.out_proj.bias, std=0.1)
        net.eval()
        proj = {
            "tap": 51,
            "W": torch.eye(d),
            "mean_in": torch.zeros(d),
            "std_in": torch.ones(d),
            "mean_out": torch.zeros(d),
            "std_out": torch.ones(d),
            "sink_out": torch.full((d,), 77.0),
            "seq_input": "base",
            "seq_config": json.dumps(net.config),
        }
        proj.update(seq_connector.state_dict_prefixed(net, dtype=torch.float32))
        x1 = torch.randn(1, 6, d)
        x2 = x1.clone()
        x2[0, 0] = 9999.0  # modifying token 0 must not leak into tokens 1..5
        o1 = clipproj.project(x1, proj, {})
        o2 = clipproj.project(x2, proj, {})
        self.assertEqual(tuple(o1.shape), (1, 6, d))
        self.assertTrue(torch.allclose(o1[0, 0], torch.full((d,), 77.0)))
        self.assertTrue(torch.allclose(o1[0, 1:], o2[0, 1:], atol=1e-5))

    def test_byte_spans_and_causal_last_n_to_1_and_ref2va_valid_mask(self):
        # F2 + F4: byte-span alignment handles split multi-byte tokens and causal last-token N:1
        src_h = torch.tensor([[1.0, 0.0], [3.0, 0.0], [10.0, 20.0], [50.0, 60.0]], dtype=torch.float32)
        out_last = ninfer_clip.unpack_by_spans(src_h, [2, 1, 3, 4], [3, 3, 2, 2], n_to_1="last")
        self.assertEqual(tuple(out_last.shape), (4, 2))
        self.assertTrue(torch.allclose(out_last[0], torch.tensor([3.0, 0.0])))     # last of ['<d', '>']
        self.assertTrue(torch.allclose(out_last[1], torch.tensor([10.0, 20.0])))   # exact 3-byte char
        self.assertTrue(torch.allclose(out_last[2], torch.tensor([50.0, 60.0])))   # 1:N first half
        self.assertTrue(torch.allclose(out_last[3], torch.tensor([50.0, 60.0])))   # 1:N second half

        # F3: ref2va uncompacted valid_mask keeps vision tokens isolated from text Conv1D/attention
        torch.manual_seed(23)
        d = 16
        net = seq_connector.BidirectionalSeqConnector(
            d_in=d, d_out=d, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, conv_kernel=5, dropout=0.0
        )
        torch.nn.init.normal_(net.out_proj.weight, std=0.1)
        torch.nn.init.normal_(net.out_proj.bias, std=0.1)
        net.eval()
        proj = {
            "tap": 51,
            "W": torch.eye(d),
            "mean_in": torch.zeros(d),
            "std_in": torch.ones(d),
            "mean_out": torch.zeros(d),
            "std_out": torch.ones(d),
            "sink_out": torch.full((d,), 5.0),
            "W_vis": torch.eye(d) * 2.0,
            "mean_in_vis": torch.zeros(d),
            "std_in_vis": torch.ones(d),
            "mean_out_vis": torch.zeros(d),
            "std_out_vis": torch.ones(d),
            "clamp_lo": torch.full((d,), -100.0),
            "clamp_hi": torch.full((d,), 100.0),
            "seq_input": "base",
            "seq_config": json.dumps(net.config),
        }
        proj.update(seq_connector.state_dict_prefixed(net, dtype=torch.float32))
        tags = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        h1 = torch.randn(1, 9, d)
        h2 = h1.clone()
        h2[0, 2:6] = 999.0  # changing vision rows must NEVER leak into text seq refinement!
        y1 = ninfer_clip.project_with_modality(h1, proj, tags, {})
        y2 = ninfer_clip.project_with_modality(h2, proj, tags, {})
        self.assertTrue(torch.allclose(y1[0, 0], torch.full((d,), 5.0)))
        self.assertTrue(torch.allclose(y1[0, tags == 1], y2[0, tags == 1], atol=1e-5))
        self.assertTrue(torch.allclose(y1[0, tags == 0], h1[0, tags == 0] * 2.0, atol=1e-5))

    def test_qi21_build_parts_multi_image_spacing_and_empty_prompt(self):
        img = torch.zeros(1, 16, 16, 3)
        p1 = ninfer_clip.build_parts("replace X with Y", images=[img], picture_label="<image%d>")
        self.assertEqual([p["type"] for p in p1], ["text", "image_url", "text"])
        self.assertEqual(p1[0]["text"], "<image1>")
        self.assertEqual(p1[2]["text"], "replace X with Y")

        p2 = ninfer_clip.build_parts("combine them", images=[img, img], picture_label="<image%d>")
        self.assertEqual([p["type"] for p in p2], ["text", "image_url", "text", "image_url", "text"])
        self.assertEqual(p2[0]["text"], "<image1>")
        self.assertEqual(p2[2]["text"], " <image2>")
        self.assertEqual(p2[4]["text"], "combine them")

        p_empty = ninfer_clip.build_parts("", images=[img], picture_label="<image%d>")
        self.assertEqual([p["type"] for p in p_empty], ["text", "image_url"])
        self.assertEqual(p_empty[0]["text"], "<image1>")


if __name__ == "__main__":
    unittest.main()
