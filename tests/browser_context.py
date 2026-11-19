"""Exercise editor context menus, timeline edits, and viewer controls on a running QA server.

Run python tests/browser_context.py http://127.0.0.1:8188/h3take --channel msedge
with Playwright installed. The fixture uses a new, disposable project.
"""

import argparse
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from uuid import uuid4
import wave

from playwright.sync_api import expect, sync_playwright


def request(url, method, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(url, method=method, data=payload, headers={"Content-Type": "application/json"}), timeout=30) as response:
        return json.load(response)


def exercise(page, base, folder):
    page.add_init_script("localStorage.setItem('h3take.project'," + json.dumps(folder) + ")")
    page.goto(base + "/editor")
    picture = page.locator("#laneV .block[data-item]").first
    audio = page.locator("#laneA .wave[data-item]").first
    menu = page.locator("#contextMenu:popover-open")
    picture.wait_for()
    for target, action in [(picture, "Extend forward"), (audio, "Mute this placement"),
                           (page.locator("#ruler"), "Mark In here"), (page.locator("#laneX"), "Add VFX here"),
                           (page.locator(".cast-head"), "Add reference")]:
        target.click(button="right", position={"x": 10, "y": 6})
        expect(menu.get_by_role("menuitem", name=action, exact=True)).to_be_visible()
        page.keyboard.press("Escape")

    audio.focus()
    page.keyboard.press("Shift+F10")
    menu.get_by_role("menuitem", name="Mute this placement").click()
    audio.focus()
    page.keyboard.press("Shift+F10")
    expect(menu.get_by_role("menuitem", name="Restore placement volume")).to_be_visible()
    page.keyboard.press("Escape")
    expect(audio).to_be_focused()

    page.locator("#shortcuts").click()
    page.locator("#keysTitle").click(button="right")
    expect(page.locator("#keys #contextMenu:popover-open")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("#keys")).to_be_visible()
    page.keyboard.press("Escape")

    page.get_by_role("tab", name="Project").click()
    format_section = page.locator('#inspBody details[data-section="Format"]')
    heading = format_section.locator("summary")
    heading.focus()
    page.keyboard.press("Space")
    expect(format_section).not_to_have_attribute("open", "")
    page.locator("#clipsToggle").click()
    page.reload()
    expect(format_section).to_be_visible()
    expect(format_section).not_to_have_attribute("open", "")
    expect(page.locator("#clips")).to_be_hidden()
    expect(page.locator("#importClip")).to_be_visible()
    heading.click(button="right")
    menu.get_by_role("menuitem", name="Expand section").click()
    expect(format_section).to_have_attribute("open", "")

    for width, height in [(1440, 900), (1024, 640), (390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        page.locator("#ruler").click(button="right", position={"x": 8, "y": 6})
        expect(menu.get_by_role("menuitem", name="Mark Out here")).to_be_visible()
        bounds = menu.bounding_box()
        assert 0 <= bounds["x"] <= width - bounds["width"] + 1
        assert 0 <= bounds["y"] <= height - bounds["height"] + 1
        page.keyboard.press("Escape")


def exercise_style_markers(page, base, folder):
    project_url = base + "/api/project/" + folder
    page.set_viewport_size({"width": 1440, "height": 900})
    page.reload()
    expect(page.locator("#laneV .block").first).to_be_visible()
    page.locator("#addStyleMarker").click()
    editor = page.locator(".style-marker-editor")
    expect(editor.get_by_label("Source frame")).to_have_value("0")
    editor.get_by_label("Marker name").fill("Cool dusk")
    editor.get_by_label("Source frame").fill("24")
    editor.get_by_label("Style prompt").fill("Blue hour, warm windows")
    editor.get_by_role("button", name="Save marker").click()
    marker = page.get_by_role("button", name="Cool dusk, frame 24", exact=True)
    expect(marker).to_be_visible()
    marker.click(button="right")
    menu = page.locator("#contextMenu:popover-open")
    menu.get_by_role("menuitem", name="Edit style marker", exact=True).click()
    expect(editor.get_by_label("Marker name")).to_be_focused()
    expect(editor.get_by_label("Style prompt")).to_have_value("Blue hour, warm windows")
    box = marker.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 30, y, steps=6)
    page.mouse.up()
    moved = page.locator("#laneStyle .style-marker")
    expect(moved).not_to_have_attribute("data-style-frame", "24")
    expect(editor.get_by_label("Source frame")).to_have_value(moved.get_attribute("data-style-frame"))
    moved.click(button="right")
    with page.expect_response(lambda response: response.request.method == "PUT"
                              and response.url.split("?")[0] == project_url and response.ok
                              and not response.request.post_data_json["clips"][0].get("style_markers")):
        menu.get_by_role("menuitem", name="Remove style marker", exact=True).click()
    expect(moved).to_have_count(0)
    assert not request(project_url, "GET")["clips"][0].get("style_markers")


def exercise_viewer_controls(page, footage):
    page.set_viewport_size({"width": 1440, "height": 900})
    dock = page.locator("#viewerToolsDock")
    more = page.get_by_role("button", name="More viewer controls")
    panel = page.get_by_role("dialog", name="Viewer tools")
    expect(dock.get_by_role("button", name="Mark In", exact=True)).to_be_visible()
    page.locator("#splitInspector").press("ArrowLeft")
    page.locator("#splitInspector").press("ArrowLeft")
    expect(more).to_be_visible()  # Only the pane changed size.

    page.locator("#toStart").click()
    page.locator("#forward").click()
    expect(page.locator("#time")).to_have_text("00:00:01")
    more.press("Enter")
    mark_in = panel.get_by_role("button", name="Mark In", exact=True)
    expect(mark_in).to_be_focused()
    mark_in.press("Enter")
    page.keyboard.press("Escape")
    expect(more).to_be_focused()
    page.locator("#forward").click()
    more.click()
    panel.get_by_role("button", name="Mark Out", exact=True).click()
    expect(panel.locator("#rangeLabel")).to_have_text("00:00:01 – 00:00:02")
    volume = panel.get_by_role("slider", name="Viewer volume")
    volume.press("Home")
    volume.press("ArrowRight")
    expect(volume).to_have_attribute("aria-valuetext", "5%")

    page.set_viewport_size({"width": 1920, "height": 900})
    expect(panel).to_be_hidden()
    expect(dock.locator("#rangeLabel")).to_have_text("00:00:01 – 00:00:02")
    expect(dock.get_by_role("slider", name="Viewer volume")).to_have_value("0.05")
    dock.get_by_role("button", name="Clear In and Out").click()
    expect(dock.locator("#rangeLabel")).to_be_hidden()

    for width, height in [(390, 844), (320, 568)]:
        page.set_viewport_size({"width": width, "height": height})
        page.get_by_role("button", name="Viewer", exact=True).click()
        expect(more).to_be_visible()
        strip = page.locator(".transport")
        assert strip.bounding_box()["height"] <= 56
        assert strip.evaluate("el => el.scrollWidth <= el.clientWidth")
        more.click()
        expect(panel.get_by_role("button", name="Add frame note")).to_be_visible()
        if width == 320:
            panel.get_by_role("button", name="Go to start").click()
            expect(page.locator("#time")).to_have_text("00:00:00")
        else:
            page.keyboard.press("Escape")
        expect(panel).to_be_hidden()

    page.get_by_role("button", name="Clips and cast", exact=True).click()
    if page.locator("#clips").is_hidden():
        page.locator("#clipsToggle").click()
    page.locator('#clips button.clip[data-clip="' + footage + '"]').click()
    page.get_by_role("button", name="Viewer", exact=True).click()
    page.wait_for_function("document.querySelector('#screen video.show')?.readyState >= 2")
    more.click()
    panel.get_by_role("button", name="Add frame note").click()
    note = page.get_by_role("dialog", name="Write a note on this frame")
    expect(note.get_by_role("button", name="Save note")).to_be_in_viewport()
    note.get_by_role("textbox").fill("Frame note across sizes")
    page.set_viewport_size({"width": 1440, "height": 1000})
    expect(page.locator("#screen #noteText")).to_have_value("Frame note across sizes")
    page.set_viewport_size({"width": 390, "height": 844})
    more.click()
    page.keyboard.press("Escape")
    expect(panel).to_be_hidden()
    expect(note).to_be_visible()
    note.get_by_role("button", name="Cancel").click()
    expect(note).to_be_hidden()


def exercise_waveform(page, base, folder):
    project_url = base + "/api/project/" + folder
    request(project_url + "/ops", "POST", {"ops": [
        {"op": "set_project", "fields": {"sizes": {"final": "448x256"}}},
    ]})
    fixture = Path(__file__).resolve().parents[1] / "docs/media/realism/neon-rain.mp4"
    imported = page.request.post(project_url + "/import", multipart={
        "file": {"name": "footage.mp4", "mimeType": "video/mp4", "buffer": fixture.read_bytes()},
    })
    assert imported.ok, imported.text()
    item = imported.json()["item"]

    tone = io.BytesIO()
    with wave.open(tone, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(2 * 24000))
    uploaded = page.request.post(project_url + "/asset?sound=1", multipart={
        "file": {"name": "sound.wav", "mimeType": "audio/wav", "buffer": tone.getvalue()},
    })
    assert uploaded.ok, uploaded.text()
    request(project_url + "/ops", "POST", {"ops": [
        {"op": "add_sound", "asset": uploaded.json(), "seconds": 1, "fields": {"name": "Timeline sound"}},
    ]})

    page.set_viewport_size({"width": 1440, "height": 900})
    page.reload()
    menu = page.locator("#contextMenu:popover-open")
    def saving():
        return page.expect_response(lambda response: response.request.method == "PUT"
                                    and response.url.split("?")[0] == project_url and response.ok)

    for selector, kind, fade, direction in [
        ('#laneA .wave[data-item="' + item + '"]', "timeline", "fade_in", 1),
        (".lane.extra .wave.sound", "sounds", "fade_out", -1),
    ]:
        waveform = page.locator(selector)
        expect(waveform).not_to_have_class(re.compile(r"\bon\b"))
        expect(waveform.locator("[data-fade]")).to_have_count(2)
        bounds = waveform.bounding_box()
        start = waveform.locator('[data-fade="fade_in"]')
        end = waveform.locator('[data-fade="fade_out"]')
        assert abs(start.bounding_box()["x"] - bounds["x"]) <= 1
        assert abs(end.bounding_box()["x"] + end.bounding_box()["width"] - bounds["x"] - bounds["width"]) <= 1

        handle = waveform.locator('[data-fade="' + fade + '"]')
        handle.scroll_into_view_if_needed()
        grip = handle.bounding_box()
        x, y = grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2
        with saving():
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + direction * bounds["width"] / 4, y, steps=6)
            page.mouse.up()
        def row():
            rows = request(project_url, "GET")[kind]
            return next(row for row in rows if row["id"] == item) if kind == "timeline" else rows[0]

        assert row()[fade] > 0
        with saving():
            waveform.click(position={"x": bounds["width"] / 2, "y": bounds["height"] / 2}, modifiers=["Alt"])
        point = waveform.locator("[data-gain-frame]")
        expect(point).to_have_count(1)
        point.click(button="right")
        with saving():
            menu.get_by_role("menuitem", name="Remove volume point", exact=True).click()
        expect(point).to_have_count(0)
        expect(waveform.locator("[data-fade]")).to_have_count(2)
        assert not row().get("gain_points") and row()[fade] > 0
        handle.click(button="right")
        with saving():
            menu.get_by_role("menuitem", name="Reset " + fade.replace("_", " "), exact=True).click()
        assert not row().get(fade)
        expect(waveform.locator("[data-fade]")).to_have_count(2)

    track = page.get_by_role("button", name="Track volume for Sound 2", exact=True)
    track.click()
    editor = page.locator('[data-automation-track="1"]')
    expect(editor).to_be_visible()
    fader = page.get_by_role("spinbutton", name="Track level in decibels")
    fader.fill("-18")
    with saving():
        fader.press("Tab")
    bounds = editor.bounding_box()
    with saving():
        editor.click(position={"x": bounds["width"] / 3, "y": bounds["height"] / 2}, modifiers=["Alt"])
    saved = request(project_url, "GET")
    track_state = saved["audio_tracks"]["1"]
    assert abs(track_state["gain"] - 10 ** (-18 / 20)) < 1e-5
    assert len(track_state["gain_points"]) == 1
    assert track_state["gain_points"][0]["frame"] > saved["sounds"][0]["frames"]
    request(project_url + "/ops", "POST", {"ops": [
        {"op": "update_sound", "sound": saved["sounds"][0]["id"], "fields": {"start": 12}},
    ]})
    page.reload()
    expect(track).to_be_visible()
    assert request(project_url, "GET")["audio_tracks"]["1"] == track_state
    track.click()
    expect(fader).to_have_value("-18")
    expect(editor.locator("[data-gain-frame]")).to_have_count(1)
    track.click()
    expect(editor).to_have_count(0)
    return imported.json()["clip"]


def exercise_project_header(page):
    project = page.get_by_role("group", name="Project controls")
    workspace = page.get_by_role("group", name="Workspace controls")
    for width in (1101, 390):
        page.set_viewport_size({"width": width, "height": 844})
        top, lower = project.bounding_box(), workspace.bounding_box()
        assert top["y"] + top["height"] <= lower["y"] + 1
        assert lower["x"] + lower["width"] <= width
        expect(project.get_by_role("button", name="Project", exact=True)).to_be_visible()
        expect(workspace.get_by_role("button", name="Video", exact=True)).to_be_visible()
    workspace.get_by_role("button", name="Sprites", exact=True).click()
    with page.context.expect_page() as opened:
        workspace.get_by_role("link", name="Sprite test room").click()
    room = opened.value
    expect(room.locator("#status")).to_have_text("Calibration marker")
    expect(room.locator("#facing option")).to_have_count(8)
    room.close()
    workspace.get_by_role("button", name="Video", exact=True).click()


def exercise_sprite_animation(page, project_url):
    request(project_url + "/ops", "POST", {"ops": [{"op": "add_cast", "fields": {
        "name": "Herald", "refmod": {"file": "qa-character.safetensors"}}}]})
    page.reload()
    page.locator("#spritesMode").click()
    page.get_by_role("button", name="Herald", exact=True).click()
    page.get_by_role("button", name="Add animation…").click()
    form = page.get_by_role("dialog", name="Add animation")
    name = form.get_by_label("Animation name")
    name.fill("Run")
    form.get_by_label("Motion prompt").fill("Spreads both wings, then folds them.")
    form.get_by_role("button", name="Add animation", exact=True).click()
    assert not name.evaluate("el => el.validity.valid")
    name.fill("Wing stretch")
    form.get_by_role("button", name="Add animation", exact=True).click()
    select = page.get_by_label("Animation", exact=True)
    expect(select).to_have_value("Wing stretch")
    # Stop at the GPU boundary. Project saves and clip creation use the real server.
    page.route("**/render", lambda route: route.fulfill(status=400, json={"error": "QA render boundary"}))
    def draft(direction):
        page.locator("#spriteCompass").get_by_role("button", name=direction + " empty", exact=True).click()
        with page.expect_response(lambda r: r.url.endswith("/render")):
            page.get_by_role("button", name="New " + direction + " draft").click()
    draft("S")
    page.get_by_role("button", name="Edit prompt…").click()
    form = page.get_by_role("dialog", name="Edit animation prompt")
    prompt = "Raise both wings, pause, then return to rest."
    form.get_by_label("Motion prompt").fill(prompt)
    form.get_by_role("button", name="Save prompt").click()
    draft("NE")
    project = request(project_url, "GET")
    clips = [c for c in project["clips"] if (c.get("sprite") or {}).get("animation") == "Wing stretch"]
    assert {c["sprite"]["direction"] for c in clips} == {"S", "NE"}
    assert all(c["sprite"]["motion"] == "prompt" and c["action"] == prompt for c in clips)
    assert project["sprite_workspaces"][clips[0]["cast"][0]]["animations"] == [{"id": "Wing stretch", "prompt": prompt}]
    page.reload()
    page.locator("#spritesMode").click()
    page.get_by_role("button", name="Herald", exact=True).click()
    expect(select).to_have_value("Wing stretch")
    assert page.locator("#spritesRoom").evaluate("el => el.scrollWidth <= el.clientWidth + 1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="QA server URL ending in /h3take")
    parser.add_argument("--channel", help="Installed Chromium browser channel, for example msedge")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    folder = request(base + "/api/projects", "POST", {"name": "Menu QA " + uuid4().hex})["folder"]
    project_url = base + "/api/project/" + folder
    try:
        request(project_url + "/ops", "POST", {"ops": [{"op": "add_clip", "place": True, "fields": {"name": "First shot"}}]})
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=args.channel)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            exercise(page, base, folder)
            exercise_style_markers(page, base, folder)
            footage = exercise_waveform(page, base, folder)
            exercise_viewer_controls(page, footage)
            exercise_project_header(page)
            exercise_sprite_animation(page, project_url)
            assert not errors, errors
            browser.close()
    finally:
        request(project_url, "DELETE")
    print("Browser context menus, style markers, waveform, and responsive viewer controls passed.")


if __name__ == "__main__":
    main()
