"""
strategy/executor.py
────────────────────
Drives planner routes in the real pymunk simulator, so every route the
planner reports is checked against full physics (collisions, pushing
elements, the Autonomous Line walls, mechanism timings).

    res = run_auton({"red1": actions1, "red2": actions2})
    res = run_skills(actions)
"""

import math
import os
from typing import Dict, List, Optional

from config.hyperparameters import CONTROL_DT
from config import game_rules as GR


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class RouteFollower:
    """Turns a list of planner macro-actions into per-step sim actions."""

    GOAL_CLEAR = 9.5        # keep the robot centre this far from goal posts

    def __init__(self, sim, robot, actions: List[dict]):
        self.sim = sim
        self.robot = robot
        self.actions = list(actions)
        self.i = 0
        self.phase = "start"
        self.t_action = 0.0
        self.waypoints = []
        self.log = []          # (action index, status, time)
        self.trace = []        # robot centre positions
        self.flips_sent = False
        self.score_stage = 0
        self.done = False
        self._last_pos = None
        self._stuck_t = 0.0

    # ── object lookup ──────────────────────────────────────────────────
    def _sim_obj(self, oid: int):
        n_pins = len(GR.PIN_STARTS)
        if oid < n_pins:
            return self.sim.pins[oid]
        return self.sim.cups[oid - n_pins]

    def _goal(self, gid):
        return next(g for g in self.sim.goals if g.goal_id == gid)

    def _toggle(self, tid):
        return next(t for t in self.sim.toggles if t.toggle_id == tid)

    # ── low-level driving ──────────────────────────────────────────────
    def _drive_point(self, px, py, mode, poi_offset, tol, slow=14.0):
        """Return (cmd, arrived).  mode: 'front' | 'rear' | 'center'.
        poi_offset = distance of the controlled point from the centre."""
        r = self.robot
        x, y = r.body.position
        h = r.body.angle
        dx, dy = px - x, py - y
        dist = math.hypot(dx, dy)
        to = math.atan2(dy, dx)
        if mode == "rear":
            want = to + math.pi
            sign = -1.0
        elif mode == "front":
            want = to
            sign = 1.0
        else:
            if abs(_wrap(to - h)) <= math.pi / 2:
                want, sign = to, 1.0
            else:
                want, sign = to + math.pi, -1.0
        remaining = dist - poi_offset
        if remaining <= tol:
            return {"left": 0.0, "right": 0.0}, True
        e = _wrap(want - h)
        thresh = 0.55 if remaining > 14 else 0.3
        if abs(e) > thresh:
            w = max(0.35, min(0.9, abs(e) * 0.9))
            w = w if e > 0 else -w
            return {"left": -w, "right": w}, False
        v = max(0.28, min(1.0, remaining / slow)) * sign
        steer = max(-0.5, min(0.5, 1.6 * e))
        return {"left": v - steer, "right": v + steer}, False

    def _obstacles(self, ignore_goal=None, ignore_obj=None):
        """Goal posts plus standing cups that still hold a pin (touching one
        knocks it over and throws the pin out)."""
        obs = [(g.x, g.y, self.GOAL_CLEAR) for g in self.sim.goals if g.goal_id != ignore_goal]
        for c in self.sim.cups:
            if c is ignore_obj or c.carried_by is not None or c.scored:
                continue
            if c.contains_pin is None or c.orientation != "vertical":
                continue
            x, y = c.body.position
            if x < 6 or x > 138 or y < 6 or y > 138:
                continue                     # wall cups: unavoidable, robot approaches square-on
            obs.append((x, y, 10.0))
        return obs

    def _plan_waypoints(self, tx, ty, reach, mode, ignore_goal=None, ignore_obj=None):
        """Detour around obstacles sitting on the straight path (up to 3)."""
        x, y = self.robot.body.position
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        ex, ey = tx - ux * reach, ty - uy * reach          # centre destination
        obs = self._obstacles(ignore_goal, ignore_obj)
        path = [(x, y), (ex, ey)]
        for _ in range(3):
            hit = None
            for k in range(len(path) - 1):
                (x0, y0), (x1, y1) = path[k], path[k + 1]
                sx, sy = x1 - x0, y1 - y0
                L2 = sx * sx + sy * sy
                if L2 < 1:
                    continue
                for ox, oy, clr in obs:
                    u = ((ox - x0) * sx + (oy - y0) * sy) / L2
                    if not (0.02 < u < 0.98):
                        continue
                    cx, cy = x0 + sx * u, y0 + sy * u
                    if math.hypot(ox - cx, oy - cy) < clr and math.hypot(ox - x1, oy - y1) > clr:
                        hit = (k, u, ox, oy, clr, sx, sy, cx, cy)
                        break
                if hit:
                    break
            if not hit:
                break
            k, u, ox, oy, clr, sx, sy, cx, cy = hit
            L = math.hypot(sx, sy)
            nx, ny = -sy / L, sx / L
            side = 1.0 if ((cx - ox) * nx + (cy - oy) * ny) >= 0 else -1.0
            wx = min(max(ox + nx * side * (clr + 3), 9), 135)
            wy = min(max(oy + ny * side * (clr + 3), 9), 135)
            path.insert(k + 1, (wx, wy))
        return path[1:-1]

    # ── main step ──────────────────────────────────────────────────────
    def step(self, dt: float) -> Optional[dict]:
        r = self.robot
        self.trace.append((round(r.body.position.x, 1), round(r.body.position.y, 1)))
        if self.done or self.i >= len(self.actions):
            self.done = True
            return {"left": 0.0, "right": 0.0}
        a = self.actions[self.i]
        typ = a["type"]
        self.t_action += dt
        limit = 4.5 if a["type"] != "park" else 6.0
        if self.t_action > limit:
            self._finish("timeout")
            return {"left": 0.0, "right": 0.0}

        reach = r.robot_length / 2.0 + 2.0

        # stuck detection → small wiggle
        p = (r.body.position.x, r.body.position.y, r.body.angle)
        if (self._last_pos is not None
                and math.hypot(p[0] - self._last_pos[0], p[1] - self._last_pos[1]) < 0.05
                and abs(p[2] - self._last_pos[2]) < 0.01):
            self._stuck_t += dt
        else:
            self._stuck_t = 0.0
        self._last_pos = p

        # un-jam: no motion for a while → drive straight away from the
        # nearest wall / goal post for a moment, then carry on
        if getattr(self, "_unjam", 0.0) > 0:
            self._unjam -= dt
            return dict(self._unjam_cmd)
        scoring_now = (typ == "place" and self.phase == "score")
        waiting = typ in ("toggle", "load") or scoring_now
        if self._stuck_t > 0.3 and not waiting:
            self._stuck_t = 0.0
            self._unjam = 0.3
            x, y = p[0], p[1]
            blockers = [(x, -1.0), (x, GR.FIELD_HEIGHT + 1), (-1.0, y), (GR.FIELD_WIDTH + 1, y)]
            blockers += [(g.x, g.y) for g in self.sim.goals]
            bx, by = min(blockers, key=lambda b: math.hypot(b[0] - x, b[1] - y))
            p = (x, y)
            hx, hy = math.cos(r.body.angle), math.sin(r.body.angle)
            v = 0.7 if ((x - bx) * hx + (y - by) * hy) >= 0 else -0.7
            self._unjam_cmd = {"left": v, "right": v}
            return dict(self._unjam_cmd)

        if self.phase == "start":
            # A 15" square robot cannot spin next to a goal post (its corner
            # hits the post), so first back off any post we are touching.
            near = [g for g in self.sim.goals
                    if math.hypot(g.x - r.body.position.x, g.y - r.body.position.y) < 12.5]
            if near and self.t_action < 0.6:
                g = near[0]
                hx, hy = math.cos(r.body.angle), math.sin(r.body.angle)
                away = (r.body.position.x - g.x) * hx + (r.body.position.y - g.y) * hy
                v = 0.6 if away >= 0 else -0.6
                return {"left": v, "right": v}
            self.t_action = 0.0
            self.flips_sent = False
            self.score_stage = 0
            if typ in ("pick_front", "pick_rear"):
                o = self._sim_obj(a["obj"])
                tgt = o.body.position
                self.waypoints = self._plan_waypoints(tgt.x, tgt.y, reach, None, o)
            elif typ == "place":
                g = self._goal(a["goal"])
                self.waypoints = self._plan_waypoints(g.x, g.y, reach, g.goal_id)
            elif typ == "toggle":
                t = self._toggle(a["toggle"])
                px, py = self._toggle_point(t)
                self.waypoints = self._plan_waypoints(px, py, 0, None)
            elif typ == "load":
                self.waypoints = self._plan_waypoints(a["loader"][0], a["loader"][1], 0, None)
            elif typ == "park":
                self.waypoints = self._plan_waypoints(a["x"], a["y"], 0, None)
            self.phase = "travel"

        # flips can happen while driving
        cmd_extra = {}
        if typ == "place" and r.mech_busy <= 0:
            # one flip at a time: each flip keeps the mechanism busy
            if not self.flips_sent:
                self.flips_sent = set()
            if a.get("flip_pin") and r.carrying_pin is not None and "pin" not in self.flips_sent:
                cmd_extra["flip_pin"] = True
                self.flips_sent.add("pin")
            elif a.get("flip_cup") and r.carrying_cup is not None and "cup" not in self.flips_sent:
                cmd_extra["flip_cup"] = True
                self.flips_sent.add("cup")

        # waypoint following
        if self.waypoints:
            wx, wy = self.waypoints[0]
            mode = "rear" if typ == "place" else ("front" if typ == "pick_front" else
                                                  "rear" if typ == "pick_rear" else "center")
            cmd, ok = self._drive_point(wx, wy, mode, 0.0, 3.0, slow=8.0)
            if ok:
                self.waypoints.pop(0)
            cmd.update(cmd_extra)
            return cmd

        if typ == "pick_front":
            o = self._sim_obj(a["obj"])
            if o.carried_by == r.robot_id:
                self._finish("ok")
                return {"left": 0.0, "right": 0.0}
            if o.carried_by is not None or o.scored:
                self._finish("taken")
                return {"left": 0.0, "right": 0.0}
            tgt = o.body.position
            cmd, _ = self._drive_point(tgt.x, tgt.y, "front", reach, 0.5, slow=10.0)
            fx, fy = r.front_point()
            if math.hypot(fx - tgt.x, fy - tgt.y) <= GR.INTAKE_RADIUS - 1:
                cmd["intake"] = True
                cmd["intake_only"] = [o]
            return cmd

        if typ == "pick_rear":
            o = self._sim_obj(a["obj"])
            pin = o.contains_pin if (a.get("pull_nested") and hasattr(o, "contains_pin")) else o
            if a.get("pull_nested"):
                if r.carrying_pin is not None and o.contains_pin is None:
                    self._finish("ok")
                    return {"left": 0.0, "right": 0.0}
                if o.contains_pin is None:
                    self._finish("taken")
                    return {"left": 0.0, "right": 0.0}
            elif o.carried_by == r.robot_id:
                self._finish("ok")
                return {"left": 0.0, "right": 0.0}
            tgt = o.body.position
            cmd, _ = self._drive_point(tgt.x, tgt.y, "rear", reach, 0.5, slow=10.0)
            bx, by = r.rear_point()
            if math.hypot(bx - tgt.x, by - tgt.y) <= GR.INTAKE_RADIUS - 1:
                cmd["intake_rear"] = True
                cmd["intake_only"] = [o]
            return cmd

        if typ == "place":
            g = self._goal(a["goal"])
            bx, by = r.rear_point()
            d_rear = math.hypot(bx - g.x, by - g.y)
            if self.phase == "travel":
                cmd, ok = self._drive_point(g.x, g.y, "rear", reach - 1.0, 0.8, slow=12.0)
                cmd.update(cmd_extra)
                cmd.update(cmd_extra)
                in_range = d_rear <= GR.SCORING_RADIUS - 3.0
                if ok or (in_range and (self._stuck_t > 0.12 or d_rear < 3.0)):
                    self.phase = "score"
                return cmd
            # scoring: one item at a time, robot stopped (after any flips)
            pending = ((a.get("flip_pin") and "pin" not in (self.flips_sent or ()) and r.carrying_pin is not None) or
                       (a.get("flip_cup") and "cup" not in (self.flips_sent or ()) and r.carrying_cup is not None))
            if pending:
                cmd = {"left": 0.0, "right": 0.0}
                cmd.update(cmd_extra)
                return cmd
            items = a["items"]
            if self.score_stage >= len(items):
                self._finish("ok")
                return {"left": 0.0, "right": 0.0}
            if r.mech_busy > 0:
                return {"left": 0.0, "right": 0.0}
            item = items[self.score_stage]
            held = r.carrying_pin if item == "pin" else r.carrying_cup
            if held is None:
                self.score_stage += 1
                return {"left": 0.0, "right": 0.0}
            before = len(g.stack)
            cmd = {"left": 0.0, "right": 0.0,
                   "score_pin" if item == "pin" else "score_cup": True}
            self._pending_check = (g, before, item)
            self.score_stage += 1
            return cmd

        if typ == "toggle":
            t = self._toggle(a["toggle"])
            if t.owner == a["colour"]:
                self._finish("ok")
                return {"left": -0.3, "right": -0.3} if False else {"left": 0.0, "right": 0.0}
            px, py = self._toggle_point(t)
            cmd, ok = self._drive_point(px, py, "center", 0.0, 2.5, slow=10.0)
            if ok or math.hypot(r.body.position.x - t.x, r.body.position.y - t.y) <= GR.TOGGLE_INTERACTION_RANGE - 3:
                cmd = {"left": 0.0, "right": 0.0, "toggle": True}
            return cmd

        if typ == "load":
            lx, ly = a["loader"]
            want_pin = any(p != "cup" for p in a["what"].split("+"))
            want_cup = "cup" in a["what"].split("+")
            got = ((not want_pin or r.carrying_pin is not None) and
                   (not want_cup or r.carrying_cup is not None))
            if got and self.t_action > 0.1:
                self._finish("ok")
                return {"left": 0.0, "right": 0.0}
            cmd, ok = self._drive_point(lx, ly, "center", 0.0, 2.0, slow=8.0)
            if self.sim._is_in_loading_zone(r, r.alliance):
                cmd = {"left": 0.0, "right": 0.0, "match_load": a["what"]}
            return cmd

        if typ == "park":
            cmd, ok = self._drive_point(a["x"], a["y"], "center", 0.0, 1.5, slow=10.0)
            from simulation.rules_engine import RulesEngine
            if ok or RulesEngine.is_robot_in_midfield(r) and math.hypot(
                    r.body.position.x - a["x"], r.body.position.y - a["y"]) < 6:
                self._finish("ok")
                self.done = True
                return {"left": 0.0, "right": 0.0}
            return cmd

        self._finish("unknown")
        return {"left": 0.0, "right": 0.0}

    def _toggle_point(self, t):
        d = 11.0
        if t.x < 5:   return t.x + d, t.y
        if t.x > 139: return t.x - d, t.y
        if t.y < 5:   return t.x, t.y + d
        return t.x, t.y - d

    def _finish(self, status: str):
        self.log.append({"i": self.i, "type": self.actions[self.i]["type"],
                         "status": status, "t": round(self.sim.time_elapsed, 2),
                         "planned_t": self.actions[self.i].get("t_end")})
        self.i += 1
        self.phase = "start"
        self.t_action = 0.0
        self.waypoints = []


# ═══════════════════════════════════════════════════════════════════════════
def _make_sim(skills: bool, render: bool, profiles=None):
    if render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from simulation.rules_engine import RulesEngine
    RulesEngine.VERBOSE = False
    from simulation.simulator import OverrideSimulator
    return OverrideSimulator(headless=not render, skills_mode=skills,
                             robot_profiles=profiles or {})


def run_auton(plans: Dict[str, List[dict]], profiles=None, render_path: str = None,
              video_path: str = None) -> dict:
    """Run the 15 s Autonomous Period with the given red/blue robot plans."""
    sim = _make_sim(False, bool(render_path or video_path), profiles)
    sim.timer_started = True
    followers = {r.robot_id: RouteFollower(sim, r, plans[r.robot_id])
                 for r in sim.robots if r.robot_id in plans}
    frames = _maybe_recorder(sim, video_path)
    while sim.match_phase == "autonomous":
        acts = []
        for r in sim.robots:
            f = followers.get(r.robot_id)
            acts.append(f.step(CONTROL_DT) if f else None)
        sim.step(CONTROL_DT, acts)
        _frame(sim, frames, followers)
    res = {
        "red_auton": getattr(sim, "_auton_score_red", 0),
        "blue_auton": getattr(sim, "_auton_score_blue", 0),
        "awp": sim.auton_awp,
        "logs": {k: f.log for k, f in followers.items()},
        "traces": {k: f.trace for k, f in followers.items()},
        "stacks": {g.label: g.stack_desc() for g in sim.goals},
        "toggles": {t.toggle_id: t.owner for t in sim.toggles},
    }
    _finish_media(sim, frames, followers, render_path, video_path)
    return res


def run_skills(actions: List[dict], profile=None, render_path: str = None,
               video_path: str = None, duration: float = GR.SKILLS_SECONDS) -> dict:
    sim = _make_sim(True, bool(render_path or video_path),
                    {"red1": profile} if profile else None)
    sim.timer_started = True
    f = RouteFollower(sim, sim.robots[0], actions)
    followers = {"red1": f}
    frames = _maybe_recorder(sim, video_path)
    while not sim.match_over:
        sim.step(CONTROL_DT, [f.step(CONTROL_DT)])
        _frame(sim, frames, followers)
    res = {"score": sim.red_score, "logs": {"red1": f.log}, "traces": {"red1": f.trace},
           "stacks": {g.label: g.stack_desc() for g in sim.goals},
           "toggles": {t.toggle_id: t.owner for t in sim.toggles}}
    _finish_media(sim, frames, followers, render_path, video_path)
    return res


# ── rendering helpers ──────────────────────────────────────────────────────
def _maybe_recorder(sim, video_path):
    if not video_path:
        return None
    try:
        from evaluation.video_recorder import VideoRecorder
        rec = VideoRecorder(video_path, sim.screen_w, sim.screen_h, fps=20)
        return rec
    except Exception as e:     # pragma: no cover
        print("[executor] video disabled:", e)
        return None


def _draw_traces(sim, followers):
    import pygame
    colours = {"red1": (255, 90, 90), "red2": (255, 170, 60),
               "blue1": (90, 150, 255), "blue2": (90, 230, 230)}
    sc = sim.render_scale
    for rid, f in followers.items():
        pts = [(x * sc, y * sc) for x, y in f.trace[::2]]
        if len(pts) > 1:
            pygame.draw.lines(sim.screen, colours.get(rid, (255, 255, 255)), False, pts, 3)


def _frame(sim, rec, followers):
    if rec is None or sim.screen is None:
        return
    import pygame
    sim.render()
    _draw_traces(sim, followers)
    frame = pygame.surfarray.array3d(sim.screen).transpose(1, 0, 2)
    rec.write_frame(frame)


def _finish_media(sim, rec, followers, render_path, video_path):
    if sim.screen is not None:
        import pygame
        sim.render()
        _draw_traces(sim, followers)
        if render_path:
            os.makedirs(os.path.dirname(render_path) or ".", exist_ok=True)
            pygame.image.save(sim.screen, render_path)
    if rec is not None:
        try:
            rec.close()
        except Exception:
            try:
                rec.release()
            except Exception:
                pass
