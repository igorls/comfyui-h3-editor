# Styles within one clip (experimental)

Style markers cross between separately encoded looks while one H3 clip is
generated. They affect generation, so an existing take needs rendering again.
They are not a playback effect or a dissolve between finished videos.

In the editor, put the playhead inside a generated clip and press **Style
marker**. The clip's **Timeline** tab lists each marker's source frame, library
look and editable style text. The **Look** lane shows markers at their places
on the cut, accounting for trim and speed. Clips reused on the timeline share
the same markers because they share the generated take.

A marker is the destination of the preceding transition. For a 124-frame
clip, a marker at frame 0 for Pencil and another at frame 123 for Clay blend
smoothly from Pencil to Clay over that shot. Add a later marker with the same
style to hold a look before changing it. A clip with its first marker later
than frame 0 starts with the project's look. There are up to 32 markers and
four distinct styles, including that starting look.

For a node workflow, set `style_markers` on **H3 Take**:

```json
[
  {"frame": 0, "style": "Monochrome graphite pencil drawing"},
  {"frame": 123, "style": "Colorful three-dimensional clay animation"}
]
```

Keep the shot's action in `prompt`. Connect H3 Take's `positive` output to
**H3 Style Guider (experimental)**, with the same model used by the sampler.
Connect its `GUIDER` output to **SamplerCustomAdvanced**. The editor wires this
automatically for both the first pass and a final's refine pass. With no
markers the editor keeps its original Basic Guider path.

Each distinct style gets a complete encoding of the same shot, cast and speech.
At every sampling step, the guider evaluates those conditionings separately
and blends their video predictions at each H3 temporal latent position with
smoothstep weights. It respects H3's uneven temporal grid. The audio prediction
comes from the first style, and held-frame masks still apply. Repeated styles
reuse an encoding. This uses ComfyUI's [custom guider extension](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_custom_sampler.py).

Expect extra render time: two distinct styles require two model evaluations
per sampling step. They run sequentially to preserve H3's batch-size-one
requirement. Encodings can have different token counts, and references and
guides are carried to each one. Direct node users can put `__h3_style__` in a
structured prompt to replace that placeholder with each style; otherwise the
style prefixes the prompt.

This is an experiment, not a guarantee that arbitrary looks will blend cleanly.
H3's temporal attention can let one style influence nearby frames, and a large
style change can also change objects or composition. Try a small draft first.

## Validation on 2026-11-17

Three 124-frame 640×352 drafts were rendered with HyperFlow H3 Flash INT8,
four Euler steps, seed 424242 and the same teapot-orbit shot: Pencil only,
Clay only, and Pencil to Clay. All completed. The blend visibly moved from
graphite shading to colored sculpted texture while the teapot kept moving.

Mean RGB chroma (maximum minus minimum channel, averaged over each frame)
provides a simple check for this deliberately monochrome-to-color example:

| Take | Frame 0 | Frame 41 | Frame 82 | Frame 123 |
|---|---:|---:|---:|---:|
| Pencil control | 2.59 | 2.98 | 2.92 | 2.98 |
| Clay control | 137.82 | 125.10 | 128.10 | 126.74 |
| Pencil to Clay draft | 2.47 | 45.52 | 117.43 | 123.16 |
| Finished at Large (1344×768) | 2.42 | 50.28 | 116.12 | 120.51 |

This supports the visible color transition in this example. It does not score
style fidelity or establish general visual quality. Reproduce the three-shot
comparison with `python tools/test_style_blend.py`; it retains a new QA project
and takes for inspection. Unit tests cover normalized temporal weights,
different-length encodings, guide propagation and preservation of the first
audio prediction. The blended draft also completed **Finish draft** at Large
(1344×768), retaining all 124 frames and the visible transition. This exercised
the style guider in the refine pass and the Small-to-Large finish path.

A second test used three distinct looks in one shot: a rowboat moved past a
lighthouse as the picture changed from graphite to layered cut paper to clay.
Markers at frames 0, 87 and 174 produced visible changes in both the 640×352
draft and its 1344×768 finish. Both takes retained all 175 frames. Recreate
the draft with `python tools/test_style_blend.py --three-styles --seconds 7`,
then use **Finish draft** to test refinement on the same latents.

Both renders were also placed as independent timeline clips. An export using
frames 0–86 of the draft followed by frames 87–174 of the finish retained all
175 frames at 24 fps without a gap.
