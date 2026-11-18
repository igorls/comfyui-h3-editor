"""Exercise editor context menus on a running QA server.

Run python tests/browser_context.py http://127.0.0.1:8188/h3take --channel msedge
with Playwright installed. The fixture uses a new, disposable project.
"""

import argparse
import json
from urllib.request import Request, urlopen
from uuid import uuid4

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
            assert not errors, errors
            browser.close()
    finally:
        request(project_url, "DELETE")
    print("Browser context menu checks passed.")


if __name__ == "__main__":
    main()
