"""
ui/menu.py
────────────────────────────────────────────────────────────────────────────
Interactive pygame launcher for the VEX Override Simulator.

Single entry point: `show_menu()` opens a window, lets the user configure
any combination of match settings (mode, per-robot controller, skills
start position, duration, matches, video recording), and returns a
`MatchConfig` describing the choice — or None if the user quit.

Design goals
------------
• ONE command to launch (no CLI args needed for normal play)
• Cover every meaningful combination: 1v1, 2v1, 1v2, 2v2 (heuristic /
  policy / human / idle per slot) and solo skills from any of the 4
  starting corners
• Visually styled to match the in-game field (dark slate + white tape
  + red/blue alliance accents)

Screens
-------
  MAIN         — Match / Skills / Custom / Exit
  MATCH_CONFIG — 4-robot grid + duration / matches / video
  SKILLS_CONFIG — start corner picker + controller + duration / matches / video
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


# ════════════════════════════════════════════════════════════════════════════
# CONFIG OBJECT
# ════════════════════════════════════════════════════════════════════════════

# Controller options per robot slot.
CTRL_HUMAN     = "human"
CTRL_HEURISTIC = "heuristic"
CTRL_POLICY    = "policy"
CTRL_IDLE      = "idle"

ALL_CTRLS_MATCH  = [CTRL_HUMAN, CTRL_HEURISTIC, CTRL_POLICY, CTRL_IDLE]
ALL_CTRLS_SKILLS = [CTRL_HUMAN, CTRL_HEURISTIC, CTRL_POLICY]

# Only red1 / blue1 can be HUMAN — the in-sim keyboard handler routes
# WASD to red1 and arrow keys to blue1, with no third controller slot.
HUMAN_SLOTS = ("red1", "blue1")


def _ctrls_for_slot(rid: str) -> List[str]:
    """Valid controller options for a given match slot."""
    if rid in HUMAN_SLOTS:
        return ALL_CTRLS_MATCH
    return [c for c in ALL_CTRLS_MATCH if c != CTRL_HUMAN]


@dataclass
class MatchConfig:
    """Everything needed to launch a run.  Built by the menu, consumed by
    the unified `run_configured_match()` in main.py."""
    mode: str = "match"           # "match" | "skills"
    controllers: Dict[str, str] = field(default_factory=lambda: {
        "red1":  CTRL_HUMAN,
        "red2":  CTRL_HEURISTIC,
        "blue1": CTRL_HEURISTIC,
        "blue2": CTRL_HEURISTIC,
    })
    skills_start: str   = "red1"        # which corner the skills bot spawns in
    duration: float     = 120.0         # seconds per match
    matches: int        = 1             # back-to-back matches
    record_video: bool  = False
    video_path: str     = "artifacts/videos/menu_run.mp4"
    headless: bool      = False         # skip rendering; run as fast as possible
    policy_path: Optional[str] = None   # required if any slot uses CTRL_POLICY

    def human_count(self) -> int:
        return sum(1 for v in self.controllers.values() if v == CTRL_HUMAN)


@dataclass
class RunResult:
    """One completed run, for the post-session summary screen."""
    index: int                   # 1-based run number
    mode: str                    # "match" | "skills"
    red_score: int     = 0
    blue_score: int    = 0
    skills_score: int  = 0
    winner: str        = ""      # "RED" | "BLUE" | "TIE"  (match only)
    sim_seconds: float = 0.0     # sim time elapsed inside the run
    wall_seconds: float = 0.0    # real time the run took (interesting headless)
    sc5b_yellow_visible: int = 0
    sc5b_credit: int   = 0       # +pts SC5b awarded at match end


# ════════════════════════════════════════════════════════════════════════════
# COLORS / FONT
# ════════════════════════════════════════════════════════════════════════════
BG          = (18, 22, 32)
PANEL       = (28, 34, 48)
PANEL_HI    = (40, 50, 70)
TEXT        = (230, 232, 240)
TEXT_DIM    = (140, 148, 165)
TAPE        = (240, 240, 245)
RED         = (220, 70, 80)
RED_HI      = (255, 110, 120)
BLUE        = (70, 130, 230)
BLUE_HI     = (110, 170, 255)
GOLD        = (255, 210, 90)
NEUTRAL     = (160, 165, 180)
BTN_HI      = (90, 130, 200)
BTN_GO      = (60, 180, 120)
BTN_GO_HI   = (100, 220, 150)
BTN_BAD     = (210, 80, 80)

WIN_W = 960
WIN_H = 600


CTRL_LABEL = {
    CTRL_HUMAN:     "HUMAN",
    CTRL_HEURISTIC: "HEURISTIC BOT",
    CTRL_POLICY:    "TRAINED POLICY",
    CTRL_IDLE:      "IDLE",
}
CTRL_COLOR = {
    CTRL_HUMAN:     GOLD,
    CTRL_HEURISTIC: (130, 220, 160),
    CTRL_POLICY:    (200, 140, 230),
    CTRL_IDLE:      NEUTRAL,
}


# ════════════════════════════════════════════════════════════════════════════
# BUTTON / WIDGET HELPERS
# ════════════════════════════════════════════════════════════════════════════
class Button:
    """Rectangular clickable area with hover/active visual states."""

    def __init__(self, rect, label, color=PANEL_HI, hi_color=BTN_HI,
                 text_color=TEXT, font=None, payload=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.hi_color = hi_color
        self.text_color = text_color
        self.font = font
        self.payload = payload
        self.hover = False

    def update(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)

    def draw(self, surf):
        c = self.hi_color if self.hover else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=8)
        pygame.draw.rect(surf, TAPE if self.hover else (60, 70, 90),
                         self.rect, 2, border_radius=8)
        if self.font and self.label:
            t = self.font.render(self.label, True, self.text_color)
            surf.blit(t, t.get_rect(center=self.rect.center))


def _draw_field_backdrop(surf):
    """Subtle field-style backdrop: dark base + faint grid + a translucent
    midfield diamond, so the menu feels like part of the same game."""
    surf.fill(BG)
    grid_color = (28, 34, 48)
    step = 32
    for x in range(0, WIN_W, step):
        pygame.draw.line(surf, grid_color, (x, 0), (x, WIN_H), 1)
    for y in range(0, WIN_H, step):
        pygame.draw.line(surf, grid_color, (0, y), (WIN_W, y), 1)
    # Diamond hint, centered
    cx, cy = WIN_W // 2, WIN_H // 2
    R = 90
    pts = [(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy)]
    pygame.draw.polygon(surf, (24, 30, 44), pts)
    pygame.draw.polygon(surf, (40, 50, 70), pts, 2)


def _draw_title(surf, fonts, subtitle="ROBOT STRATEGY LAB"):
    title = fonts["title"].render("VEX  OVERRIDE", True, TEXT)
    sub   = fonts["small"].render(subtitle, True, TEXT_DIM)
    surf.blit(title, title.get_rect(center=(WIN_W // 2, 60)))
    surf.blit(sub,   sub.get_rect(center=(WIN_W // 2, 96)))
    # Accent bar
    pygame.draw.rect(surf, RED,  (WIN_W // 2 - 200, 110, 200, 3))
    pygame.draw.rect(surf, BLUE, (WIN_W // 2,        110, 200, 3))


def _draw_footer(surf, fonts, hint):
    foot = fonts["small"].render(hint, True, TEXT_DIM)
    surf.blit(foot, foot.get_rect(center=(WIN_W // 2, WIN_H - 18)))


def _draw_toggle(surf, fonts, rect, mouse_pos, label_on, label_off,
                  is_on, hi_color=GOLD):
    """Draw a labeled toggle box.  Glows with `hi_color` when on."""
    hov = rect.collidepoint(mouse_pos)
    bg  = PANEL_HI if hov else PANEL
    border = hi_color if is_on else (60, 70, 90)
    pygame.draw.rect(surf, bg, rect, border_radius=6)
    pygame.draw.rect(surf, border, rect, 2, border_radius=6)
    txt = label_on if is_on else label_off
    col = hi_color if is_on else TEXT_DIM
    t = fonts["small"].render(txt, True, col)
    surf.blit(t, t.get_rect(center=rect.center))


# ════════════════════════════════════════════════════════════════════════════
# MAIN MENU SCREEN
# ════════════════════════════════════════════════════════════════════════════
def _screen_main(screen, fonts, config: MatchConfig) -> str:
    """Show the main menu.  Returns the chosen next-screen key:
       'match' | 'skills' | 'quickstart' | 'quit'."""
    clock = pygame.time.Clock()

    bw, bh = 360, 64
    cx = WIN_W // 2 - bw // 2
    y0 = 180
    gap = 16

    buttons = [
        Button((cx, y0,                bw, bh),
               "MATCH  —  2v2 (configurable)",
               color=PANEL, hi_color=PANEL_HI,
               font=fonts["med"], payload="match"),
        Button((cx, y0 + (bh + gap),   bw, bh),
               "SKILLS  —  Solo 60s run",
               color=PANEL, hi_color=PANEL_HI,
               font=fonts["med"], payload="skills"),
        Button((cx, y0 + 2*(bh + gap), bw, bh),
               "QUICK  —  watch bots play",
               color=PANEL, hi_color=PANEL_HI,
               font=fonts["med"], payload="quickstart"),
        Button((cx, y0 + 3*(bh + gap), bw, bh),
               "QUIT",
               color=PANEL, hi_color=BTN_BAD,
               font=fonts["med"], payload="quit"),
    ]

    while True:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return "quit"
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for b in buttons:
                    if b.rect.collidepoint(ev.pos):
                        return b.payload
        for b in buttons:
            b.update(mouse)

        _draw_field_backdrop(screen)
        _draw_title(screen, fonts)
        for b in buttons:
            b.draw(screen)
        _draw_footer(screen, fonts,
                     "Click to select  •  ESC to quit  •  Hover for highlight")
        pygame.display.flip()
        clock.tick(60)


# ════════════════════════════════════════════════════════════════════════════
# MATCH CONFIG SCREEN
# ════════════════════════════════════════════════════════════════════════════
def _screen_match_config(screen, fonts,
                         config: MatchConfig) -> Optional[bool]:
    """Configure a 2v2 match: cycle each slot's controller, set match
    options.  Returns True to start, False to go back, None to quit."""
    clock = pygame.time.Clock()

    # Slot card geometry
    card_w, card_h = 220, 110
    gap_x, gap_y = 30, 22
    grid_left = WIN_W // 2 - card_w - gap_x // 2
    grid_top  = 150

    # Slot positions: red on left column, blue on right
    slot_layout = {
        "red1":  (grid_left,                       grid_top),
        "red2":  (grid_left,                       grid_top + card_h + gap_y),
        "blue1": (grid_left + card_w + gap_x,      grid_top),
        "blue2": (grid_left + card_w + gap_x,      grid_top + card_h + gap_y),
    }
    slot_rects = {rid: pygame.Rect(x, y, card_w, card_h)
                  for rid, (x, y) in slot_layout.items()}

    # Bottom controls.  Layout per group: [-] [LABEL] [+]
    #   group width = 30 + 8 + 160 + 8 + 30 = 236
    # Two groups side-by-side with 32px gap between them.
    bottom_y = grid_top + 2 * (card_h + gap_y) + 20
    group_w  = 236
    gap_g    = 32
    total_w  = group_w * 2 + gap_g
    g1_left  = WIN_W // 2 - total_w // 2
    g2_left  = g1_left + group_w + gap_g
    minus_dur     = pygame.Rect(g1_left,                  bottom_y, 30, 30)
    dur_label_box = pygame.Rect(g1_left + 38,             bottom_y, 160, 30)
    plus_dur      = pygame.Rect(g1_left + 206,            bottom_y, 30, 30)
    minus_m       = pygame.Rect(g2_left,                  bottom_y, 30, 30)
    m_label_box   = pygame.Rect(g2_left + 38,             bottom_y, 160, 30)
    plus_m        = pygame.Rect(g2_left + 206,            bottom_y, 30, 30)
    # Two side-by-side toggles: HEADLESS and RECORD VIDEO (mutually exclusive)
    headless_box = pygame.Rect(WIN_W // 2 - 290, bottom_y + 50, 280, 30)
    video_box    = pygame.Rect(WIN_W // 2 + 10,  bottom_y + 50, 280, 30)

    back_btn  = Button((WIN_W // 2 - 220, WIN_H - 70, 180, 44),
                       "BACK",  color=PANEL, hi_color=PANEL_HI,
                       font=fonts["med"], payload="back")
    start_btn = Button((WIN_W // 2 + 40,  WIN_H - 70, 180, 44),
                       "START MATCH",
                       color=(40, 80, 60), hi_color=BTN_GO_HI,
                       text_color=(20, 30, 25), font=fonts["med"],
                       payload="start")

    while True:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Slot click cycles controller through the slot's valid set.
                # red2/blue2 can't be HUMAN (only red1/blue1 are wired to
                # WASD / arrow keys), so the cycle skips HUMAN there.
                for rid, rect in slot_rects.items():
                    if rect.collidepoint(ev.pos):
                        opts = _ctrls_for_slot(rid)
                        cur = config.controllers[rid]
                        if cur not in opts:
                            cur = opts[0]
                        config.controllers[rid] = opts[
                            (opts.index(cur) + 1) % len(opts)
                        ]
                # Right-click cycles BACKWARD — handled in MOUSEBUTTONUP below
                if minus_dur.collidepoint(ev.pos):
                    config.duration = max(15.0, config.duration - 15.0)
                if plus_dur.collidepoint(ev.pos):
                    config.duration = min(600.0, config.duration + 15.0)
                if minus_m.collidepoint(ev.pos):
                    config.matches = max(1, config.matches - 1)
                if plus_m.collidepoint(ev.pos):
                    config.matches = min(99, config.matches + 1)
                if headless_box.collidepoint(ev.pos):
                    config.headless = not config.headless
                    if config.headless:
                        config.record_video = False    # mutually exclusive
                if video_box.collidepoint(ev.pos):
                    config.record_video = not config.record_video
                    if config.record_video:
                        config.headless = False
                if back_btn.rect.collidepoint(ev.pos):
                    return False
                if start_btn.rect.collidepoint(ev.pos):
                    config.mode = "match"
                    return True
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                # Right-click on a slot cycles BACKWARD through valid set.
                for rid, rect in slot_rects.items():
                    if rect.collidepoint(ev.pos):
                        opts = _ctrls_for_slot(rid)
                        cur = config.controllers[rid]
                        if cur not in opts:
                            cur = opts[0]
                        config.controllers[rid] = opts[
                            (opts.index(cur) - 1) % len(opts)
                        ]

        back_btn.update(mouse); start_btn.update(mouse)

        # ── Draw ────────────────────────────────────────────────────
        _draw_field_backdrop(screen)
        _draw_title(screen, fonts, subtitle="MATCH  CONFIGURATION")

        # Alliance headers
        red_h  = fonts["med"].render("RED ALLIANCE",  True, RED_HI)
        blue_h = fonts["med"].render("BLUE ALLIANCE", True, BLUE_HI)
        screen.blit(red_h,  red_h.get_rect(center=(grid_left + card_w // 2,
                                                   grid_top - 22)))
        screen.blit(blue_h, blue_h.get_rect(center=(grid_left + card_w + gap_x + card_w // 2,
                                                    grid_top - 22)))

        # Slot cards
        for rid, rect in slot_rects.items():
            ctrl = config.controllers[rid]
            base = (60, 30, 36) if rid.startswith("red") else (28, 44, 80)
            hi   = RED if rid.startswith("red") else BLUE
            hov  = rect.collidepoint(mouse)
            c    = (base[0] + 20, base[1] + 20, base[2] + 20) if hov else base
            pygame.draw.rect(screen, c, rect, border_radius=10)
            pygame.draw.rect(screen, hi, rect, 2, border_radius=10)

            # Slot label
            slabel = fonts["med"].render(rid.upper(), True, TEXT)
            screen.blit(slabel, (rect.x + 14, rect.y + 10))
            # Controller chip
            chip_col = CTRL_COLOR[ctrl]
            ctxt = fonts["big"].render(CTRL_LABEL[ctrl], True, chip_col)
            screen.blit(ctxt, ctxt.get_rect(midleft=(rect.x + 14, rect.y + 60)))
            # Hint
            hint = fonts["tiny"].render("click=next  •  right-click=prev",
                                        True, TEXT_DIM)
            screen.blit(hint, hint.get_rect(midbottom=(rect.centerx,
                                                      rect.bottom - 8)))

        # Bottom strip — duration / matches / video.  Each group is
        # rendered as [-] [LABEL_BOX] [+] with the label centered in
        # its own box so it never overlaps the +/- buttons.
        for rect, sym in [(minus_dur, "-"), (plus_dur, "+"),
                          (minus_m, "-"), (plus_m, "+")]:
            hov = rect.collidepoint(mouse)
            pygame.draw.rect(screen, PANEL_HI if hov else PANEL,
                             rect, border_radius=4)
            t = fonts["med"].render(sym, True, TEXT)
            screen.blit(t, t.get_rect(center=rect.center))
        # Label boxes (read-only background panel so the text reads cleanly)
        pygame.draw.rect(screen, PANEL,        dur_label_box, border_radius=4)
        pygame.draw.rect(screen, (60, 70, 90), dur_label_box, 1, border_radius=4)
        pygame.draw.rect(screen, PANEL,        m_label_box,   border_radius=4)
        pygame.draw.rect(screen, (60, 70, 90), m_label_box,   1, border_radius=4)
        dur_label = fonts["small"].render(
            f"Duration: {int(config.duration)} s", True, TEXT)
        screen.blit(dur_label, dur_label.get_rect(center=dur_label_box.center))
        m_label = fonts["small"].render(
            f"Matches: {config.matches}", True, TEXT)
        screen.blit(m_label, m_label.get_rect(center=m_label_box.center))

        # Headless + Record-video toggles (mutually exclusive)
        _draw_toggle(screen, fonts, headless_box, mouse,
                     "HEADLESS: ON  (fast sim)", "Headless: off",
                     config.headless, hi_color=(140, 230, 200))
        _draw_toggle(screen, fonts, video_box, mouse,
                     "RECORD VIDEO: ON", "Record video: off",
                     config.record_video, hi_color=GOLD)

        back_btn.draw(screen); start_btn.draw(screen)
        _draw_footer(screen, fonts,
                     "Click a slot to cycle controller  •  ESC = back")
        pygame.display.flip()
        clock.tick(60)


# ════════════════════════════════════════════════════════════════════════════
# SKILLS CONFIG SCREEN
# ════════════════════════════════════════════════════════════════════════════
_SKILL_STARTS = ["red1", "red2", "blue1", "blue2"]
_SKILL_START_LABEL = {
    "red1":  "NW (10, 72)",
    "red2":  "N  (72, 10)",
    "blue1": "S  (72, 136)",
    "blue2": "SE (136, 72)",
}
# Skills-mode controllers — IDLE makes no sense here (would just sit there).
_SKILL_CTRLS = [CTRL_HUMAN, CTRL_HEURISTIC, CTRL_POLICY]


def _draw_mini_field(screen, fonts, x, y, w, h, highlight_start):
    """Draw a tiny field diagram with the 4 starting corners; the chosen
    one is highlighted."""
    pygame.draw.rect(screen, (40, 48, 64), (x, y, w, h), border_radius=8)
    pygame.draw.rect(screen, (80, 90, 110), (x, y, w, h), 2, border_radius=8)
    # Field is 144 x 144 in sim units; corners (approximated)
    corner_uv = {
        "red1":  (10/144,  72/144),
        "red2":  (72/144,  10/144),
        "blue1": (72/144,  136/144),
        "blue2": (136/144, 72/144),
    }
    for key, (u, v) in corner_uv.items():
        px = int(x + u * w)
        py = int(y + v * h)
        is_red = key.startswith("red")
        is_hi  = (key == highlight_start)
        r  = 11 if is_hi else 7
        c  = (RED_HI if is_red else BLUE_HI) if is_hi else \
             (RED   if is_red else BLUE)
        pygame.draw.circle(screen, c, (px, py), r)
        if is_hi:
            pygame.draw.circle(screen, TAPE, (px, py), r + 4, 2)
    # Midfield diamond hint
    cx = x + w // 2; cy = y + h // 2
    R = min(w, h) // 5
    pygame.draw.polygon(screen, (60, 70, 90),
                        [(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy)], 2)
    cap = fonts["tiny"].render("field overview", True, TEXT_DIM)
    screen.blit(cap, cap.get_rect(center=(x + w // 2, y + h - 10)))


def _screen_skills_config(screen, fonts,
                          config: MatchConfig) -> Optional[bool]:
    clock = pygame.time.Clock()
    bottom_y = 380

    # Default skills duration is 60s, not 120
    if config.duration == 120.0:
        config.duration = 60.0
    if config.mode != "skills":
        config.mode = "skills"

    # Controller selector
    ctrl_card = pygame.Rect(WIN_W // 2 - 360, 160, 260, 110)
    # 4 start-corner cards
    start_card_w = 110
    start_card_h = 80
    start_cards_total_w = 4 * start_card_w + 3 * 12
    start_y = 160
    start_x0 = WIN_W // 2 - 60
    start_rects = {}
    for i, key in enumerate(_SKILL_STARTS):
        rx = start_x0 + i * (start_card_w + 12)
        start_rects[key] = pygame.Rect(rx, start_y, start_card_w, start_card_h)

    # Mini field
    mini_x = WIN_W // 2 - 60
    mini_y = 260
    mini_w = start_cards_total_w
    mini_h = 110

    # Bottom strip.  Same [-] [LABEL] [+] layout as the match screen.
    group_w  = 236
    gap_g    = 32
    total_w  = group_w * 2 + gap_g
    g1_left  = WIN_W // 2 - total_w // 2
    g2_left  = g1_left + group_w + gap_g
    minus_dur     = pygame.Rect(g1_left,        bottom_y, 30, 30)
    dur_label_box = pygame.Rect(g1_left + 38,   bottom_y, 160, 30)
    plus_dur      = pygame.Rect(g1_left + 206,  bottom_y, 30, 30)
    minus_m       = pygame.Rect(g2_left,        bottom_y, 30, 30)
    m_label_box   = pygame.Rect(g2_left + 38,   bottom_y, 160, 30)
    plus_m        = pygame.Rect(g2_left + 206,  bottom_y, 30, 30)
    headless_box  = pygame.Rect(WIN_W // 2 - 290, bottom_y + 50, 280, 30)
    video_box     = pygame.Rect(WIN_W // 2 + 10,  bottom_y + 50, 280, 30)

    back_btn  = Button((WIN_W // 2 - 220, WIN_H - 70, 180, 44),
                       "BACK", color=PANEL, hi_color=PANEL_HI,
                       font=fonts["med"], payload="back")
    start_btn = Button((WIN_W // 2 + 40,  WIN_H - 70, 180, 44),
                       "START SKILLS",
                       color=(40, 80, 60), hi_color=BTN_GO_HI,
                       text_color=(20, 30, 25), font=fonts["med"],
                       payload="start")

    while True:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if ctrl_card.collidepoint(ev.pos):
                    cur = config.controllers.get("red1", CTRL_HEURISTIC)
                    cur = cur if cur in _SKILL_CTRLS else CTRL_HEURISTIC
                    nxt = _SKILL_CTRLS[
                        (_SKILL_CTRLS.index(cur) + 1) % len(_SKILL_CTRLS)
                    ]
                    config.controllers["red1"] = nxt
                for key, rect in start_rects.items():
                    if rect.collidepoint(ev.pos):
                        config.skills_start = key
                if minus_dur.collidepoint(ev.pos):
                    config.duration = max(15.0, config.duration - 15.0)
                if plus_dur.collidepoint(ev.pos):
                    config.duration = min(180.0, config.duration + 15.0)
                if minus_m.collidepoint(ev.pos):
                    config.matches = max(1, config.matches - 1)
                if plus_m.collidepoint(ev.pos):
                    config.matches = min(99, config.matches + 1)
                if headless_box.collidepoint(ev.pos):
                    config.headless = not config.headless
                    if config.headless:
                        config.record_video = False
                if video_box.collidepoint(ev.pos):
                    config.record_video = not config.record_video
                    if config.record_video:
                        config.headless = False
                if back_btn.rect.collidepoint(ev.pos):
                    return False
                if start_btn.rect.collidepoint(ev.pos):
                    return True

        back_btn.update(mouse); start_btn.update(mouse)

        # ── Draw ────────────────────────────────────────────────────
        _draw_field_backdrop(screen)
        _draw_title(screen, fonts, subtitle="SKILLS  CONFIGURATION")

        # Controller card
        ctrl = config.controllers.get("red1", CTRL_HEURISTIC)
        if ctrl not in _SKILL_CTRLS:
            ctrl = CTRL_HEURISTIC
            config.controllers["red1"] = ctrl
        hov = ctrl_card.collidepoint(mouse)
        pygame.draw.rect(screen, (40, 50, 70) if hov else PANEL,
                         ctrl_card, border_radius=10)
        pygame.draw.rect(screen, RED, ctrl_card, 2, border_radius=10)
        hdr = fonts["med"].render("CONTROLLER", True, TEXT)
        screen.blit(hdr, hdr.get_rect(midtop=(ctrl_card.centerx,
                                              ctrl_card.y + 10)))
        ctrl_txt = fonts["big"].render(CTRL_LABEL[ctrl], True, CTRL_COLOR[ctrl])
        screen.blit(ctrl_txt, ctrl_txt.get_rect(center=(
            ctrl_card.centerx, ctrl_card.y + 70)))
        hint = fonts["tiny"].render("click to cycle", True, TEXT_DIM)
        screen.blit(hint, hint.get_rect(midbottom=(ctrl_card.centerx,
                                                   ctrl_card.bottom - 6)))

        # Start-corner cards
        for key, rect in start_rects.items():
            is_red = key.startswith("red")
            chosen = (key == config.skills_start)
            hov = rect.collidepoint(mouse)
            base = (60, 30, 36) if is_red else (28, 44, 80)
            c    = (base[0]+30, base[1]+30, base[2]+30) if (hov or chosen) else base
            pygame.draw.rect(screen, c, rect, border_radius=8)
            border_c = TAPE if chosen else (RED if is_red else BLUE)
            pygame.draw.rect(screen, border_c, rect, 3 if chosen else 2,
                             border_radius=8)
            lbl = fonts["small"].render(_SKILL_START_LABEL[key], True, TEXT)
            screen.blit(lbl, lbl.get_rect(center=(rect.centerx,
                                                   rect.y + 24)))
            tag = fonts["tiny"].render(key.upper(),
                                       True, RED_HI if is_red else BLUE_HI)
            screen.blit(tag, tag.get_rect(center=(rect.centerx,
                                                   rect.y + 50)))

        # Mini field
        _draw_mini_field(screen, fonts, mini_x, mini_y, mini_w, mini_h,
                         config.skills_start)

        # Bottom strip — [-] [LABEL] [+] groups, label centered in box.
        for rect, sym in [(minus_dur, "-"), (plus_dur, "+"),
                          (minus_m, "-"), (plus_m, "+")]:
            hov = rect.collidepoint(mouse)
            pygame.draw.rect(screen, PANEL_HI if hov else PANEL,
                             rect, border_radius=4)
            t = fonts["med"].render(sym, True, TEXT)
            screen.blit(t, t.get_rect(center=rect.center))
        pygame.draw.rect(screen, PANEL,        dur_label_box, border_radius=4)
        pygame.draw.rect(screen, (60, 70, 90), dur_label_box, 1, border_radius=4)
        pygame.draw.rect(screen, PANEL,        m_label_box,   border_radius=4)
        pygame.draw.rect(screen, (60, 70, 90), m_label_box,   1, border_radius=4)
        dur_label = fonts["small"].render(
            f"Duration: {int(config.duration)} s", True, TEXT)
        screen.blit(dur_label, dur_label.get_rect(center=dur_label_box.center))
        m_label = fonts["small"].render(
            f"Runs: {config.matches}", True, TEXT)
        screen.blit(m_label, m_label.get_rect(center=m_label_box.center))

        # Headless + Record-video toggles (mutually exclusive)
        _draw_toggle(screen, fonts, headless_box, mouse,
                     "HEADLESS: ON  (fast sim)", "Headless: off",
                     config.headless, hi_color=(140, 230, 200))
        _draw_toggle(screen, fonts, video_box, mouse,
                     "RECORD VIDEO: ON", "Record video: off",
                     config.record_video, hi_color=GOLD)

        back_btn.draw(screen); start_btn.draw(screen)
        _draw_footer(screen, fonts,
                     "Pick start corner + controller  •  ESC = back")
        pygame.display.flip()
        clock.tick(60)


# ════════════════════════════════════════════════════════════════════════════
# POST-SESSION SUMMARY SCREEN
# ════════════════════════════════════════════════════════════════════════════
def show_summary(results: List["RunResult"],
                 config: "MatchConfig") -> None:
    """After all runs complete, show a per-run table + grand totals.

    Re-opens / resizes the pygame window to menu dimensions (since the
    sim may have closed its own display), waits for any key / click,
    then returns control to the menu loop."""
    pygame.display.quit()
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("VEX Override -- Session Summary")
    fonts = _build_fonts()
    clock = pygame.time.Clock()

    is_skills = (config.mode == "skills")
    # Gather stats
    if is_skills:
        scores = [r.skills_score for r in results]
    else:
        scores = [r.red_score + r.blue_score for r in results]
    n = len(results) if results else 1
    mean = sum(scores) / n if results else 0.0
    smin = min(scores) if results else 0
    smax = max(scores) if results else 0
    var = sum((s - mean) ** 2 for s in scores) / n if results else 0.0
    sstd = var ** 0.5
    total_wall = sum(r.wall_seconds for r in results)
    total_sim  = sum(r.sim_seconds for r in results)
    speedup    = (total_sim / total_wall) if total_wall > 0 else 0.0

    # Stat cards: 4 cards across the top
    card_w, card_h = 200, 80
    card_gap = 16
    total_w = 4 * card_w + 3 * card_gap
    cards_x = WIN_W // 2 - total_w // 2
    cards_y = 130
    stat_cards = []
    label = "Skills mean" if is_skills else "Total mean"
    stat_cards.append((label,          f"{mean:.1f}",  GOLD))
    stat_cards.append(("Min",          f"{smin}",      (160, 220, 200)))
    stat_cards.append(("Max",          f"{smax}",      (160, 220, 200)))
    stat_cards.append(("Std dev",      f"{sstd:.1f}",  TEXT_DIM))

    # Scrollable list of runs below stat cards
    list_top = cards_y + card_h + 24
    row_h    = 24
    max_rows = (WIN_H - list_top - 110) // row_h
    scroll = [0]
    total_rows = len(results)
    max_scroll = max(0, total_rows - max_rows)

    back_btn = Button((WIN_W // 2 - 100, WIN_H - 60, 200, 44),
                      "BACK TO MENU",
                      color=(40, 80, 60), hi_color=BTN_GO_HI,
                      text_color=(20, 30, 25), font=fonts["med"],
                      payload="back")

    waiting = True
    while waiting:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                waiting = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_RETURN,
                              pygame.K_SPACE):
                    waiting = False
                if ev.key == pygame.K_DOWN:
                    scroll[0] = min(max_scroll, scroll[0] + 1)
                if ev.key == pygame.K_UP:
                    scroll[0] = max(0, scroll[0] - 1)
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and back_btn.rect.collidepoint(ev.pos):
                    waiting = False
                if ev.button == 4:
                    scroll[0] = max(0, scroll[0] - 1)
                if ev.button == 5:
                    scroll[0] = min(max_scroll, scroll[0] + 1)
        back_btn.update(mouse)

        # ── Draw ────────────────────────────────────────────────────
        _draw_field_backdrop(screen)
        kind = "SKILLS" if is_skills else "MATCH"
        sub = f"{kind}  SESSION  SUMMARY  ({len(results)} runs)"
        _draw_title(screen, fonts, subtitle=sub)

        # Stat cards
        for i, (label, value, color) in enumerate(stat_cards):
            x = cards_x + i * (card_w + card_gap)
            r = pygame.Rect(x, cards_y, card_w, card_h)
            pygame.draw.rect(screen, PANEL, r, border_radius=10)
            pygame.draw.rect(screen, color, r, 2, border_radius=10)
            l = fonts["small"].render(label, True, TEXT_DIM)
            screen.blit(l, l.get_rect(midtop=(r.centerx, r.y + 8)))
            v = fonts["title"].render(value, True, color)
            v = pygame.transform.smoothscale(
                v, (min(card_w - 16, v.get_width()),
                    min(card_h - 32, v.get_height())))
            screen.blit(v, v.get_rect(center=(r.centerx, r.y + 50)))

        # Speed indicator (headless only)
        if total_wall > 0 and config.headless:
            speed_lbl = fonts["small"].render(
                f"Sim time: {total_sim:.1f}s   Wall: {total_wall:.2f}s   "
                f"Speedup: {speedup:.1f}x",
                True, (140, 230, 200))
            screen.blit(speed_lbl, speed_lbl.get_rect(
                midtop=(WIN_W // 2, cards_y + card_h + 4)))
            list_top_eff = list_top + 14
        else:
            list_top_eff = list_top

        # Table header
        if is_skills:
            headers = [("# ", 50), ("START", 90), ("SCORE", 90),
                       ("SC5b", 80), ("SIM s", 90), ("WALL s", 90)]
        else:
            headers = [("# ", 50), ("RED", 70), ("BLUE", 70),
                       ("TOTAL", 80), ("WINNER", 90), ("WALL s", 90)]
        header_x = WIN_W // 2 - sum(w for _, w in headers) // 2
        hx = header_x
        for h, w in headers:
            t = fonts["small"].render(h, True, GOLD)
            screen.blit(t, (hx + 6, list_top_eff))
            hx += w
        pygame.draw.line(screen, (60, 70, 90),
                         (header_x, list_top_eff + 20),
                         (header_x + sum(w for _, w in headers),
                          list_top_eff + 20), 1)

        # Rows
        row_y = list_top_eff + 26
        view = results[scroll[0]:scroll[0] + max_rows]
        for r in view:
            hx = header_x
            if is_skills:
                cells = [str(r.index), r.winner or "—",
                         str(r.skills_score),
                         f"+{r.sc5b_credit}" if r.sc5b_credit else "—",
                         f"{r.sim_seconds:.1f}", f"{r.wall_seconds:.2f}"]
            else:
                cells = [str(r.index), str(r.red_score), str(r.blue_score),
                         str(r.red_score + r.blue_score),
                         r.winner or "—", f"{r.wall_seconds:.2f}"]
            for (h, w), cell in zip(headers, cells):
                # Colorize the WINNER column
                col = TEXT
                if not is_skills and h.strip() == "WINNER":
                    col = (RED_HI if cell == "RED" else
                           BLUE_HI if cell == "BLUE" else TEXT_DIM)
                # Highlight the skills SCORE column
                if is_skills and h.strip() == "SCORE":
                    col = GOLD
                t = fonts["small"].render(cell, True, col)
                screen.blit(t, (hx + 6, row_y))
                hx += w
            row_y += row_h

        # Scroll hint
        if max_scroll > 0:
            hint = fonts["tiny"].render(
                f"showing rows {scroll[0] + 1}-{scroll[0] + len(view)} "
                f"of {total_rows}   "
                "(mouse wheel / arrows to scroll)",
                True, TEXT_DIM)
            screen.blit(hint, hint.get_rect(
                center=(WIN_W // 2, row_y + 12)))

        back_btn.draw(screen)
        _draw_footer(screen, fonts,
                     "Press ENTER / ESC / click BACK to return to menu")
        pygame.display.flip()
        clock.tick(60)

    pygame.display.quit()


def _build_fonts() -> Dict[str, pygame.font.Font]:
    return {
        "title": pygame.font.SysFont("arialblack,arial", 44, bold=True),
        "med":   pygame.font.SysFont("consolas,monospace", 20, bold=True),
        "big":   pygame.font.SysFont("consolas,monospace", 22, bold=True),
        "small": pygame.font.SysFont("consolas,monospace", 16),
        "tiny":  pygame.font.SysFont("consolas,monospace", 12),
    }


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def show_menu(initial: Optional[MatchConfig] = None) -> Optional[MatchConfig]:
    """Show the launcher UI.  Returns the chosen MatchConfig, or None if
    the user quit without starting anything."""
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("VEX Override -- Launcher")
    fonts = _build_fonts()

    config = initial if initial is not None else MatchConfig()

    state = "main"
    while True:
        if state == "main":
            choice = _screen_main(screen, fonts, config)
            if choice == "quit":
                pygame.display.quit()
                pygame.quit()
                return None
            if choice == "match":
                state = "match_cfg"
            elif choice == "skills":
                state = "skills_cfg"
            elif choice == "quickstart":
                # Quick = 4 heuristic bots, no human, default settings
                config.mode = "match"
                config.controllers = {rid: CTRL_HEURISTIC
                                      for rid in ("red1","red2","blue1","blue2")}
                return config
        elif state == "match_cfg":
            result = _screen_match_config(screen, fonts, config)
            if result is None:
                pygame.display.quit(); pygame.quit(); return None
            if result is False:
                state = "main"
            else:
                return config
        elif state == "skills_cfg":
            result = _screen_skills_config(screen, fonts, config)
            if result is None:
                pygame.display.quit(); pygame.quit(); return None
            if result is False:
                state = "main"
            else:
                # Other slots irrelevant in skills mode
                return config


if __name__ == "__main__":
    cfg = show_menu()
    print("Returned config:", cfg)
