# Editing rendered takes

H3 Editor should let a person and an agent finish the same cut from the same project. A crop, an audio mix or a range export uses the renders already on disk. It should take no model inference to preview or undo an editing decision.

| Feature | Human workflow | Agent workflow | Acceptance check |
| --- | --- | --- | --- |
| Crop and reframe | Select a timeline placement, reposition or resize the crop in the viewer, and reset it when needed. | Read and set a placement's framing in normalized source coordinates. | Place one take twice. Crop one placement and compare the viewer and both exported placements. |
| Export a range | Set In and Out on the timeline and export that section. | Give the export a start frame and an exclusive end frame. | Export across a cut, a transition, a gap and a fading sound. Compare the frames and audio with the same interval of the whole export. |
| Sound fades and volume curves | Fade a sound in or out and place gain points along its duration. | Read and edit the fades and gain points, using explicit units and a documented interpolation rule. | Lower music under speech, then restore it. Listen from the middle, trim the sound, and compare playback with full and range exports. |
| Resolution controls | Choose a common or custom aspect ratio. Enter either the long edge or a megapixel budget and see the resolved dimensions. | Request the same sizing calculation for drafts and finals. | Check 3:2, 4:3, 7:5 and 21:9, portrait and landscape, H3's 32-pixel grid and old custom width/height settings. |
| Manage a clip's takes | Select unwanted takes in the inspector and remove them in one operation. | Remove specific take IDs and receive the resulting selection. | Reload the project, try a stale save, and confirm an independently placed take still plays. Keep shared render files available to other placements and project history. |

## Implementation and verification

All five features are implemented in the working tree. The editor uses timeline placements for crops and the existing In/Out marks for range exports. Open **Manage takes** in the Takes inspector to remove several versions together. Clearing all renders from a generated shot preserves its prompt and timeline edits. Imported footage keeps its final take. The [agent API](../skills/h3-editor/references/api.md) documents crop coordinates, exclusive export end frames, sound gain points, resolution targets and take removal.

An isolated ComfyUI instance imported a two-color clip, placed it twice, cropped only the second placement, added an eight-frame dissolve and mixed a tone with fades and three gain points. A 37-frame export spanning the join matched the corresponding interval in the 88-frame export to a mean RGB error below 0.002 on a 0–255 scale (measured at 96 × 72). Interior audio samples had a correlation above 0.99999. Edge also opened the crop editor, reset a placement's framing, displayed the sound controls, and prepared an export from the viewer's marks. The GPU-container suite completed 595 Python tests with one skip. All 12 JavaScript tests passed.

## GPU experiments

- **VOSR2.** Use short H3 renders with moving hair, skin and neon rain. Compare 2× output with a conventional resize and the published source frames. Compare changing the noise seed across frames with keeping it fixed. Record resolution, wall time, GPU memory and temporal behavior.
- **Mid-frame image guides.** Hold the prompt, start image, model settings and seed constant. Measure how closely the render reaches a guide at the requested frame and how motion changes on either side. Include one and two interior guides at draft and native render sizes.

[GPU experiment results](editing-render-tests.md)
