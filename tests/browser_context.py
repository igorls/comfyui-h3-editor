"""Exercise editor context menus on a running QA server.

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
            exercise_waveform(page, base, folder)
            assert not errors, errors
            browser.close()
    finally:
        request(project_url, "DELETE")
    print("Browser context menu and waveform checks passed.")


if __name__ == "__main__":
    main()
