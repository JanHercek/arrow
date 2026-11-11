#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) Jan Hercek
'''
Tests for arrow.py - the closed-form (no time step) trajectory of an arrow.

Four kinds of check, and each one earns its place by what it catches that the others cannot:
  * the TABLE and the three contract tests: the public surface must answer them exactly;
  * agreement with an independent RK4 march defined in this file - correctness of the physics,
    written from the equations rather than from the solver, so a shared mistake is unlikely;
  * the closed form's own limits: with drag_coef = 0 it must reproduce the textbook vacuum
    range to nanometres, a shot must never fall faster than sqrt(g/k), and the auto crossing
    search must answer every distance in range with the flattest shot that hits;
  * the two interfaces: the REPL and the command line must speak one grammar, answer in the order
    they are asked in, print one help screen, and refuse nonsense without killing the session;
  * the lofted route: the second solution to the same target, which `segments` cannot reach and
    `steepest_angle` can, with the forward model signing every angle it returns.

arrow.py has no TIME_STEP: that is the whole point of the file, so it is asserted here, and the
boundary cases that used to cost seconds are timed.

Run it next to arrow.py (as in the repository):  python3 arrow_test.py
'''
import ast
import contextlib
import io
import math
import os
import pathlib
import re
import subprocess
import sys
import typing as t
import unittest

sys.dont_write_bytecode = True      # running the suite must not leave __pycache__ beside the repo

import arrow


def _flat(text: str) -> str:
    '''The transcript with every run of spaces inside each line collapsed to one.

    Labels are asserted through this so a test does not also have to restate how wide the answer
    block happened to be that day. `test_the_answer_block_aligns_its_equal_signs` is what pins the
    spacing, and it pins it exactly.
    '''
    return "\n".join(" ".join(line.split()).replace("]=", "] =") for line in text.splitlines())


def _block(text: str) -> list[str]:
    '''The answer rows only: every line that prints a `[unit]` and a value.'''
    return [line for line in text.splitlines() if re.search(r"\[[^\]]*\] *=", line)]


def _rk4_range(arw: arrow.Arrow, angle: float, dt: float = 1e-4) -> float:
    '''Ground-frame RK4 of the model's ODE, stopped where the model says the flight ends.

    Written from the equations, not from the solver: the drag uses the velocity relative to the
    air, so this does not share the Galilean-shift trick arrow.py uses for wind.  The bracketing
    step is re-integrated 2000x finer so the crossing is located rather than interpolated -
    otherwise the reference's own error (~1e-6 m) is what the tolerance would be measuring.
    '''
    k: float = arw.drag_k()
    g: float = arw.gravity_accel
    w: float = arw.wind
    rising: bool = arw.segments == 1

    def deriv(s: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        rx, ry = s[2] - w, s[3]
        sp: float = math.hypot(rx, ry)
        return (s[2], s[3], -k * sp * rx, -g - k * sp * ry)

    def step(s: tuple[float, float, float, float], h: float) -> tuple[float, float, float, float]:
        q1 = deriv(s)
        q2 = deriv(tuple(s[i] + 0.5 * h * q1[i] for i in range(4)))  # type: ignore[call-overload]
        q3 = deriv(tuple(s[i] + 0.5 * h * q2[i] for i in range(4)))  # type: ignore[call-overload]
        q4 = deriv(tuple(s[i] + h * q3[i] for i in range(4)))  # type: ignore[call-overload]
        return tuple(s[i] + h / 6.0 * (q1[i] + 2 * q2[i] + 2 * q3[i] + q4[i])  # type: ignore[return-value]
                     for i in range(4))

    s: tuple[float, float, float, float] = (
        0.0, arw.archer_alt + arw.ar_launch_alt,
        arw.ar_launch_velo * math.cos(angle), arw.ar_launch_velo * math.sin(angle))
    tt: float = 0.0
    while tt < 40.0:
        prev: tuple[float, float, float, float] = s
        s = step(s, dt)
        tt += dt
        crossed = (prev[1] <= 0.0 < s[1]) if rising else (prev[1] > 0.0 >= s[1])
        if crossed:
            sub, ts, h = prev, tt - dt, dt / 2000.0
            for _ in range(2000):
                nxt = step(sub, h)
                if ((sub[1] <= 0.0 <= nxt[1]) if rising else (sub[1] > 0.0 >= nxt[1])):
                    frac = (0.0 - sub[1]) / (nxt[1] - sub[1]) if nxt[1] != sub[1] else 1.0
                    return sub[0] + frac * (nxt[0] - sub[0])
                sub, ts = nxt, ts + h
            raise AssertionError("RK4: the bracket never contained the crossing")
    raise AssertionError("RK4: the sight line was never crossed")


class ArrowTestCases(unittest.TestCase):
    '''Closed-form model: contract, physics, limits.'''
    # segments, archer_alt, angle[rad], distance[m]
    TABLE: t.Final[tuple[tuple[int, float, float, float], ...]] = (
        (2, 0., .031, 64.0),
        (2, 0., .007, 9.50),
        (2, 0., .6319, 612.0),
        (1, -10., 1.3755, 1.99),
        (2, 10., -1.3700, 2.02))

    # ---------------------------------------------------------------- contract --
    def _lands(self, arw: arrow.Arrow, ang: float, dist: float, seg: int) -> float:
        '''How far the angle lands from `dist`, checked on the crossing that answered.

        The crossing is pinned even though a forward call with segments = 0 now follows crossing:
        a test that inherits which crossing to ask about cannot tell a wrong answer from a wrong
        question, and asking about the other crossing measures the other half of the same arc.
        '''
        keep: int = arw.segments
        arw.segments = seg
        try:
            return abs(arw.get_horizontal_distance(ang) - dist)
        finally:
            arw.segments = keep

    def test_get_horizontal_distance(self) -> None:
        '''horizontal distance calculus check'''
        arw = arrow.Arrow()
        for v in self.TABLE:
            arw.segments = v[0]
            arw.archer_alt = v[1]
            self.assertAlmostEqual(v[3], arw.get_horizontal_distance(v[2]), None,
                                   f"angle={v[2]}, distance={v[3]}", 0.05)
        arw.segments = 2
        arw.archer_alt = 0.
        self.assertRaises(arrow.Unreachable, arw.get_horizontal_distance, 0.)

    def test_get_vertical_distance(self) -> None:
        '''vertical error (shooting error) check'''
        arw = arrow.Arrow()
        for v in self.TABLE:
            arw.segments = v[0]
            arw.archer_alt = v[1]
            self.assertAlmostEqual(0., arw.get_vertical_distance(v[2], v[3]), None,
                                   f"angle={v[2]}, distance={v[3]}", 0.05)
        arw.segments = 2
        arw.archer_alt = 0.
        self.assertRaises(arrow.Unreachable, arw.get_vertical_distance, 0., 900.)
        self.assertRaises(arrow.Unreachable, arw.get_vertical_distance, 0., 0.001)
        self.assertRaises(arrow.BadInput, arw.get_vertical_distance, math.pi, 64.)
        self.assertLess(arw.get_vertical_distance(0., 64.), 0.)  # flat shot drops

    def test_get_angle(self) -> None:
        '''shooting angle calculus check'''
        arw = arrow.Arrow()
        for v in self.TABLE:
            arw.segments = v[0]
            arw.archer_alt = v[1]
            self.assertAlmostEqual(v[2], arw.get_angle(v[3]), None,
                                   f"angle={v[2]}, distance={v[3]}", 0.005)
        arw.segments = 2
        arw.archer_alt = 0.
        self.assertRaises(arrow.Unreachable, arw.get_angle, 690.)

    # ------------------------------------------------------- agreement, physics --
    def test_agrees_with_rk4(self) -> None:
        '''Forward model against an independent march: mm, not cm.'''
        cases: t.Final[tuple[tuple[int, float, float], ...]] = (
            (2, 0., .031), (2, 0., .007), (2, 0., .6319), (2, 0., 1.2),
            (1, -10., 1.3755), (2, 10., -1.37), (2, -50., .35), (2, 50., -.2),
            (1, -100., 1.5271), (2, -100., .8), (2, 500., -1.3))
        for seg, alt, ang in cases:
            arw = arrow.Arrow()
            arw.segments, arw.archer_alt = seg, alt              # type: ignore[assignment]
            with self.subTest(segments=seg, archer_alt=alt, angle=ang):
                got = arw.get_horizontal_distance(ang)
                want = _rk4_range(arw, ang)
                self.assertLessEqual(abs(got - want), 0.001,
                                     f"{got:.4f} m vs RK4 {want:.4f} m")

    def test_wind_is_relative_velocity_drag(self) -> None:
        '''Drag answers to the air; the closed form's Galilean shift must match that march.'''
        for w in (-10., 0., 10., 25.):
            arw = arrow.Arrow()
            arw.wind = w
            with self.subTest(wind=w):
                got = arw.get_horizontal_distance(0.05)
                want = _rk4_range(arw, 0.05)
                self.assertLessEqual(abs(got - want), 0.001,
                                     f"wind {w}: {got:.4f} m vs RK4 {want:.4f} m")

    def test_get_angle_lands(self) -> None:
        '''Every reported angle must land inside DIST_ACCURACY of the target.'''
        cases: t.Final[tuple[tuple[int, float, float], ...]] = (
            (2, 0., 10.), (2, 0., 64.), (2, 0., 300.), (2, 0., 612.),
            (2, -50., 50.), (2, 50., 50.), (1, -10., 20.), (1, -50., 50.),
            (1, -100., 50.), (1, -100., 300.), (2, 100., 400.), (1, -10., 2.),
            (2, -100., 300.), (2, -100., 500.), (1, -100., 319.),
            (2, -50., 300.), (2, -30., 200.), (2, -80., 450.))   # the target-above rows exercise
        # the flat crossing: with the target above the archer the eye line can already overshoot
        # and the flattest hit lies flatter than the eye line, below it.
        for seg, alt, dist in cases:
            arw = arrow.Arrow()
            arw.segments, arw.archer_alt = seg, alt              # type: ignore[assignment]
            with self.subTest(segments=seg, archer_alt=alt, distance=dist):
                try:
                    ang = arw.get_angle(dist)
                except arrow.Unreachable:
                    continue                                 # an honest miss is allowed
                self.assertLessEqual(abs(arw.get_horizontal_distance(ang) - dist),
                                     arw.DIST_ACCURACY,
                                     f"get_angle({dist})={ang:.5f} lands "
                                     f"{arw.get_horizontal_distance(ang):.3f}")

    # ---------------------------------------------------------- closed-form only --
    def test_vacuum_limit_is_exact(self) -> None:
        '''No drag, arrow on the sight line: the textbook range, to nanometres.

        archer_alt cancels ar_launch_alt so the flight starts and ends at y = 0 and
        v0^2 sin(2 a)/g applies exactly; 7 places is nanometres, which is what a quadrature
        that is exact for this integrand (not merely fine) has to deliver.
        '''
        for alpha in (0.1, 0.3, 0.7):
            arw = arrow.Arrow()
            arw.drag_coef = 0.
            arw.archer_alt = -arw.ar_launch_alt
            v0, g = arw.ar_launch_velo, arw.gravity_accel
            want: float = v0 * v0 * math.sin(2. * alpha) / g
            with self.subTest(angle=alpha):
                self.assertAlmostEqual(arw.get_horizontal_distance(alpha), want, 7)

    def test_nothing_beats_terminal_speed(self) -> None:
        '''Drop from 500 m at near-vertical: the speed settles at sqrt(g/k), never above.'''
        arw = arrow.Arrow()
        arw.archer_alt = 500.
        arw.get_horizontal_distance(-1.3)
        self.assertLess(arw.hit_speed, arw.terminal_speed() * 1.02)
        self.assertAlmostEqual(arw.terminal_speed(), 96.9268, 3)

    def test_no_time_step_anywhere(self) -> None:
        '''This file exists to not march: no TIME_STEP, and no 4-second boundary cases.'''
        self.assertFalse(hasattr(arrow.Arrow, "TIME_STEP"))
        arw = arrow.Arrow()
        arw.segments, arw.archer_alt = 2, -50.                   # type: ignore[assignment]
        arw.max_dist = 700.
        import time
        t0: float = time.perf_counter()
        try:
            arw.get_angle(579.5)                       # a stepping solver needed ~4 s to say no
        except arrow.Unreachable:
            pass
        self.assertLess(time.perf_counter() - t0, 5.0, "a search this bounded must not take seconds")

    def test_the_files_own_promises_are_true(self) -> None:
        '''Three promises about the source that the README no longer states because they are not
        about archery: standard library only, no recursion, and no loop a question can spin. A
        promise nobody runs is a decoration, so the suite holds them now.'''
        tree: ast.Module = ast.parse(pathlib.Path(arrow.__file__).read_text(encoding="utf-8"))
        imports: list[str] = sorted({n.names[0].name if isinstance(n, ast.Import) else n.module
                                     for n in ast.walk(tree)
                                     if isinstance(n, (ast.Import, ast.ImportFrom))})
        self.assertEqual(imports, ["math", "sys", "typing"],
                         "the header promises nothing outside the standard library")
        funcs: dict[str, ast.FunctionDef] = {}
        bound: dict[str, set[str]] = {}
        for fn in [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)]:
            names: set[str] = ({a.arg for a in fn.args.args}
                               | {x.arg for x in fn.args.kwonlyargs})
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    names.add(node.name)
            funcs[fn.name], bound[fn.name] = fn, names
        edges: dict[str, set[str]] = {}
        for name, fn in funcs.items():
            called: set[str] = set()
            for call in ast.walk(fn):
                if isinstance(call, ast.Call):
                    if isinstance(call.func, ast.Attribute):
                        called.add(call.func.attr)
                    elif isinstance(call.func, ast.Name) and call.func.id not in bound[name]:
                        called.add(call.func.id)
            edges[name] = {c for c in called if c in funcs and c != name}
        state: dict[str, int] = {}
        cycles: list[list[str]] = []

        def walk(node: str, stack: list[str]) -> None:
            state[node] = 1
            for nxt in edges.get(node, ()):
                if state.get(nxt, 0) == 1:
                    cycles.append(stack[stack.index(nxt):] + [nxt] if nxt in stack
                                  else [node, nxt])
                elif state.get(nxt, 0) == 0:
                    walk(nxt, stack + [nxt])
            state[node] = 2

        for node in list(edges):
            if state.get(node, 0) == 0:
                walk(node, [node])
        self.assertEqual(cycles, [], "the call graph must have no cycle: no recursion anywhere")
        spun: list[str] = [f"line {n.lineno}: for ... in {ast.unparse(n.iter)}"
                           for n in ast.walk(tree) if isinstance(n, ast.For)
                           if not (isinstance(n.iter, ast.Call)
                                   and getattr(n.iter.func, "id", "") == "range")
                           and ast.unparse(n.iter) not in ("tokens", "(1, 2)", "(True, False)")]
        self.assertEqual(spun, [],
                         "every loop is range(<bound>), a literal pair, or the line the user typed")
        whiles = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
        self.assertEqual(len(whiles), 1, "one while loop in the file: the REPL's input loop")

    def test_bad_input_survives_the_auto_search(self) -> None:
        '''A setup error must surface as BadInput, not be swallowed into "cannot be hit".'''
        heavy: arrow.Arrow = arrow.Arrow()
        heavy.ar_mass = -1.
        self.assertRaises(arrow.BadInput, heavy.get_angle, 50.)
        gale: arrow.Arrow = arrow.Arrow()
        gale.wind = 400.                             # a headwind faster than the arrow
        self.assertRaises(arrow.BadInput, gale.get_angle, 50.)

    def test_refuses_what_it_cannot_answer(self) -> None:
        '''Nothing may spin, and nothing may be guessed at.'''
        for bad in (4, -1, 1.5, "two"):
            arw = arrow.Arrow()
            arw.segments = bad                                     # type: ignore[assignment]
            with self.subTest(segments=bad):
                self.assertRaises(arrow.Unreachable, arw.get_angle, 100.)
                self.assertRaises(arrow.Unreachable, arw.get_horizontal_distance, 0.1)
        arw = arrow.Arrow()
        with self.subTest(name="inputs"):
            self.assertRaises(arrow.BadInput, arw.get_horizontal_distance, 1.6)
            self.assertRaises(arrow.Unreachable, arw.get_angle, 0.)
            self.assertRaises(arrow.Unreachable, arw.get_angle, 900.)

    def test_calibrate_recovers_the_drag(self) -> None:
        '''One chronograph reading must turn back into the drag_coef that made it.'''
        truth = arrow.Arrow()
        truth.drag_coef = 0.55
        truth.get_angle(100.)                          # calibrate() reads hit_speed the same way
        speed: float = truth.hit_speed                   # the chronograph reading at 100 m
        arw = arrow.Arrow()
        got = arw.calibrate(100., speed)
        self.assertAlmostEqual(got, 0.55, 3)

    # --------------------------------------------------------------- auto segments --
    def test_auto_answers_what_either_single_crossing_answers(self) -> None:
        '''segments=0 must miss no shot that segments=1 or segments=2 can make.'''
        cases: t.Final[tuple[tuple[float, float], ...]] = (
            (0., 3.), (0., 6.), (0., 7.), (0., 7.4), (0., 8.), (0., 10.), (0., 300.),
            (0., 612.), (-10., 2.), (-10., 20.), (-50., 120.), (-100., 300.), (-100., 319.),
            (100., 400.), (-200., 400.), (50., 50.))
        for alt, dist in cases:
            auto, each = None, []
            for seg in (0, 1, 2):
                arw = arrow.Arrow()
                arw.archer_alt, arw.segments = alt, seg          # type: ignore[assignment]
                try:
                    ang = arw.get_angle(dist)
                except arrow.Unreachable:
                    ang = None
                if seg == 0:
                    auto = (ang, arw.crossing)
                else:
                    each.append(ang)
            with self.subTest(archer_alt=alt, distance=dist):
                self.assertIsNotNone(auto[0], f"auto says no, forced say {each}")
                ang, used = auto
                assert ang is not None                            # asserted above
                self.assertIn(used, (1, 2), "crossing must name the crossing that answered")
                self.assertLessEqual(self._lands(arw, ang, dist, used),
                                     arw.DIST_ACCURACY)
                for seg, forced in zip((1, 2), each):
                    if forced is not None:
                        # 1e-6 rad is 0.3 mm at 300 m: far below anything physical, and 1000x
                        # the jitter between two searches finding the same root, while a branch
                        # actually lost is off by degrees.
                        self.assertLessEqual(ang, forced + 1e-6,
                                             f"auto picked {ang:.5f}, forced segments={seg} "
                                             f"found the flatter {forced:.5f}")

    def test_auto_hands_over_at_the_rising_reach(self) -> None:
        '''The rising crossing answers the short shots the falling one refuses, and back.'''
        level: float = arrow.Arrow()
        reach: float = level.ar_launch_velo * math.sqrt(2. * -(level.archer_alt
                                                               + level.ar_launch_alt)
                                                        / level.gravity_accel)
        self.assertAlmostEqual(reach, 7.4, 1)          # the bound in _crossing_can_reach
        for dist, want in ((1., 1), (3., 1), (6., 1), (10., 2), (50., 2), (612., 2)):
            arw = arrow.Arrow()
            with self.subTest(distance=dist):
                ang = arw.get_angle(dist)
                used: int = arw.crossing
                self.assertEqual(used, want, f"{dist} m used crossing {used}")
                self.assertLessEqual(self._lands(arw, ang, dist, used), arw.DIST_ACCURACY)
        below: float = arrow.Arrow().get_angle(3.)     # 3 m level: rising, and nearly flat
        self.assertLess(abs(below), 0.02, "a 3 m shot must not be a 365 m lob")

    def test_forced_lob_is_still_reachable(self) -> None:
        '''Auto must not take away the deliberate high shot: segments=2 still lobs.'''
        flat, lob = arrow.Arrow().get_angle(120.), None
        arw = arrow.Arrow()
        arw.segments, arw.archer_alt = 2, -50.                    # type: ignore[assignment]
        lob = arw.get_angle(120.)
        self.assertLess(flat, lob)
        self.assertGreater(math.degrees(lob), 60., "the forced falling shot is the plunging one")
        self.assertEqual(arw.crossing, 2)

    # ------------------------------------------- forced crossings, notes, the wall --
    def test_forced_crossing_falls_back_and_says_so(self) -> None:
        '''Asking for a crossing that cannot reach gets the other answer plus an explanation.'''
        for seg, dist, want_s, other in ((1, 300., 2, "rising"), (2, 3., 1, "falling")):
            arw = arrow.Arrow()
            arw.segments = seg
            with self.subTest(asked=seg, distance=dist):
                ang = arw.get_angle(dist)
                self.assertEqual(arw.crossing, want_s, "the crossing that answered")
                self.assertLessEqual(self._lands(arw, ang, dist, want_s), arw.DIST_ACCURACY)
                self.assertIn(f"you asked for the {other} crossing", arw.note)
                self.assertIn("cannot hit this target", arw.note)
                self.assertIn("crossing's answer", arw.note)

    def test_forced_crossing_names_the_flatter_alternative(self) -> None:
        '''s=2 is the wall shot, so the note must hand back the flat shot it is not using.'''
        arw = arrow.Arrow()
        arw.archer_alt, arw.segments = -50., 2
        high = arw.get_angle(120.)
        self.assertEqual(arw.crossing, 2)
        self.assertIn("high shot", arw.note)
        self.assertIn("+0.45697", arw.note)                # the rising angle, quoted
        self.assertIn("1.383", arw.note)                   # and its flight time
        flat = arrow.Arrow()
        flat.archer_alt, flat.segments = -50., 1
        low = flat.get_angle(120.)
        self.assertEqual(flat.crossing, 1)
        self.assertIn("flat shot", flat.note)
        self.assertIn("+1.46834", flat.note)
        self.assertLess(low, high, "the rising crossing is the flatter shot")
        self.assertLess(flat.hit_time, 2., "the flat shot is in the air under 2 s")
        self.assertGreater(arw.hit_time, 15., "the high shot is in the air over 15 s")

    def test_auto_picks_the_faster_route(self) -> None:
        '''segments = 0 owes the quicker arrow: judged against both forced crossings.'''
        for alt, dist in ((-5., 3.), (-50., 20.), (-50., 120.), (-100., 120.),
                          (-100., 300.), (-200., 300.), (0., 50.), (0., 300.)):
            angles: dict[int, float] = {}
            times: dict[int, float] = {}
            for seg in (1, 2):
                arw = arrow.Arrow()
                arw.archer_alt, arw.segments = alt, seg
                try:
                    got_seg: float = arw.get_angle(dist)
                except arrow.Unreachable:
                    continue
                # A forced crossing now falls back to the other one, so the crossing that answered
                # has to be read back: recording a fallback as if the asked crossing had answered
                # is how "auto picked the slower route" got measured once already.
                if arw.crossing == seg:
                    angles[seg], times[seg] = got_seg, arw.hit_time
            self.assertTrue(angles, f"{dist} m at archer_alt {alt} must be hittable")
            auto = arrow.Arrow()
            auto.archer_alt = alt
            ang = auto.get_angle(dist)
            with self.subTest(archer_alt=alt, distance=dist):
                self.assertEqual(auto.crossing, min(angles, key=lambda s: times[s]),
                                 "auto must answer with the crossing in the air for less time")
                self.assertAlmostEqual(ang, angles[auto.crossing], 5)
                self.assertEqual(auto.note, "", "auto asked for nothing, so nothing to explain")

    def test_forward_call_follows_the_crossing_that_answered(self) -> None:
        '''A forward call after an auto get_angle must describe that same shot, not the other one.'''
        arw = arrow.Arrow()
        ang = arw.get_angle(3.)                        # the short level shot is the rising one
        self.assertEqual(arw.crossing, 1)
        self.assertAlmostEqual(arw.get_horizontal_distance(ang), 3., delta=arw.DIST_ACCURACY)
        self.assertEqual(arw.crossing, 1, "the forward call stayed on the crossing that answered")
        fresh = arrow.Arrow()                          # nothing asked yet: the range, so falling
        pinned = arrow.Arrow()
        pinned.segments = 2
        self.assertAlmostEqual(fresh.get_horizontal_distance(0.35),
                               pinned.get_horizontal_distance(0.35), places=9)

    def test_the_wall_recipe_clears_what_the_flat_shot_cannot(self) -> None:
        '''The point of s=2: a wall at half range that the flat shot goes under.'''
        arw = arrow.Arrow()
        arw.archer_alt, arw.segments = -50., 2
        high = arw.get_angle(120.)
        over = arw.get_vertical_distance(high, 60.)
        flat = arrow.Arrow()
        flat.archer_alt, flat.segments = -50., 1
        low = flat.get_angle(120.)
        under = flat.get_vertical_distance(low, 60.)
        self.assertGreater(over, 100., "the plunging shot passes well over a wall at half range")
        self.assertLess(under, 0., "the flat shot is still under the sight line at half range")
        self.assertGreater(over - under, 300., "same target, same bow, different arrow")

    def test_an_unflyable_probe_is_a_refusal_not_a_setup_error(self) -> None:
        '''An angle past the vertical guard is the search running out of road, not a broken bow.'''
        for seg in (0, 1, 2):
            arw = arrow.Arrow()
            arw.archer_alt, arw.segments = -200., seg      # target 200 m up, 1 cm away
            with self.subTest(segments=seg):
                with self.assertRaises(arrow.Unreachable) as got:
                    arw.get_angle(0.01)
                msg: str = str(got.exception)
                self.assertIn("cannot be hit", msg)
                self.assertIn("will not fly", msg)         # the guard quoted, not swallowed
        # an angle the *caller* hands in stays a setup error, where it always was
        self.assertRaises(arrow.BadInput, arrow.Arrow().get_horizontal_distance, 1.5707514)

    def test_the_lofted_shot_is_the_other_solution_to_the_same_target(self) -> None:
        '''Two angles hit any hittable target; segments reaches one of them, steepest_angle the other.'''
        for d in (1., 10., 60., 300., 612.):
            flat: float = arrow.Arrow().get_angle(d)              # what auto answers
            steep: float = arrow.Arrow().steepest_angle(d)
            probe: arrow.Arrow = arrow.Arrow()
            with self.subTest(distance=d):
                self.assertGreater(steep, flat + 0.01, "the lofted route is the steeper one")
                self.assertAlmostEqual(probe.get_horizontal_distance(steep), d,
                                       delta=probe.DIST_ACCURACY,
                                       msg="the model must agree its own lofted angle lands")
                self.assertEqual(probe.crossing, 2,
                                 "a level target's lofted twin is a falling crossing")
                for seg in (1, 2):          # the gap this fills: both askings give the flat shot
                    forced: arrow.Arrow = arrow.Arrow()
                    forced.segments = seg
                    self.assertAlmostEqual(forced.get_angle(d), flat, places=9)

    def test_the_lofted_shot_says_what_it_costs(self) -> None:
        '''An archer choosing between the two must be told what the other one would have done.'''
        arw: arrow.Arrow = arrow.Arrow()
        arw.steepest_angle(300.)
        for claim in ("lofted", "74.9 deg", "apex 341.5 m", "+0.17461 rad", "3.455 s", "flat route"):
            self.assertIn(claim, arw.note, f"the note does not say {claim!r}: {arw.note}")
        self.assertAlmostEqual(arw.hit_x, 300., delta=arw.DIST_ACCURACY)
        self.assertGreater(arw.hit_time, 10., "the numbers must describe the lofted shot, not the flat")
        self.assertEqual(arw.crossing, 2)

    def test_the_lofted_ask_does_not_inherit_the_last_crossing(self) -> None:
        '''An object that just answered on the rising crossing must still loft the shot.'''
        used: arrow.Arrow = arrow.Arrow()
        used.get_angle(5.0)                                 # a level 5 m shot: auto takes rising
        self.assertEqual(used.crossing, 1, "the setup of this test: auto chose the rising crossing")
        lofted: float = used.steepest_angle(5.0)
        self.assertGreater(lofted, 1.5, "the lofted ask must not search the crossing it last used")
        self.assertEqual(used.crossing, 2, "and it must report the crossing that really answered")
        self.assertAlmostEqual(used.hit_x, 5.0, delta=used.DIST_ACCURACY,
                               msg="the read-back must describe the lofted flight, not the last one")
        self.assertGreater(used.apex, 300., "an apex of a few metres is the rising shot's number")
        self.assertAlmostEqual(arrow.Arrow().get_horizontal_distance(lofted), 5.0,
                               delta=arrow.Arrow().DIST_ACCURACY)

    def test_steepest_angle_refuses_where_there_is_nothing_to_find(self) -> None:
        '''Even straight up, an arrow lands 0.12 m out: below that the lofted shot does not exist.'''
        for near in (0.03, 0.1):
            with self.subTest(distance=near), self.assertRaises(arrow.Unreachable) as ctx:
                arrow.Arrow().steepest_angle(near)
            self.assertIn("too close for the lofted shot", str(ctx.exception))
        with self.assertRaises(arrow.Unreachable) as ctx:
            arrow.Arrow().steepest_angle(-5.)
        self.assertIn("target_distance=-5.0", str(ctx.exception))
        high: arrow.Arrow = arrow.Arrow()
        high.archer_alt = -500.                                   # a target 500 m above the archer
        with self.assertRaises(arrow.Unreachable) as ctx:
            high.steepest_angle(300.)
        self.assertIn("no angle of this bow", str(ctx.exception))

    def test_s_3_asks_for_the_lofted_shot_through_get_angle_and_both_interfaces(self) -> None:
        '''One answer whether asked as a library call, a setting, a CLI line or a REPL line.'''
        want: float = arrow.Arrow().steepest_angle(60.)
        arw: arrow.Arrow = arrow.Arrow()
        arw.segments = 3
        self.assertAlmostEqual(arw.get_angle(60.), want, places=9)
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _typed(["s=3 d=60", "q"]):
            arrow.Arrow().interactive()
        text: str = _flat(out.getvalue())
        self.assertIn("output: launch angle [rad] = 1.5216", text)
        self.assertIn("segments [-] = 3", text)
        self.assertIn("crossing [-] = 2 (falling crossing)", text)
        self.assertIn("note: this is the lofted shot", text)
        # the prompt itself needs a real pipe: input() writes no prompt to a StringIO stdin
        banner: str = subprocess.run([sys.executable, arrow.__file__], input="q\n",
                                     capture_output=True, text=True, timeout=30, check=True,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).stdout
        self.assertIn(f"arrow.py({arrow.VERSION}):", banner)
        self.assertIn("Segments=0|1|2|3 (0=auto, 3=lofted)", banner,
                      f"the prompt must advertise the fourth shot: {banner}")
        cli: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(cli), _argv([sys.argv[0], "s=3", "d=60"]):
            arrow.main()
        self.assertIn("angle 1.52155 rad", cli.getvalue(),
                       "the command line must give the same lofted shot")

    # ------------------------------------------------------------------- ui -----
    def test_repl_accepts_both_spellings(self) -> None:
        '''`d=50 l=-50` and `d50 l-50` are one grammar: the = is optional, never required.'''
        lines: list[str] = []
        for typed in ("d=50 l=-50", "d50 l-50", "d=50 l-50", "D=50 L=-50".lower()):
            out: io.StringIO = io.StringIO()
            with contextlib.redirect_stdout(out), _typed([typed, "q"]):
                arrow.Arrow().interactive()
            lines.append(out.getvalue())
        for typed, text in zip(("d=50 l=-50", "d50 l-50", "d=50 l-50"), lines[:3]):
            with self.subTest(typed=typed):
                flat: str = _flat(text)
                self.assertIn("output: launch angle [rad] = 0.8103", flat)
                self.assertIn("crossing [-] = 1 (rising crossing)", flat)
        self.assertEqual(lines[0], lines[1], "the two spellings must not differ by one character")
        self.assertEqual(lines[0], lines[2])

    def test_repl_answers_in_the_order_it_is_asked(self) -> None:
        '''Question, then the state, then the answer, then the numbers - crossing leads them.'''
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _typed(["d=50 l=-50", "q"]):
            arrow.Arrow().interactive()
        text: str = _flat(out.getvalue())        # spacing is another test's business
        order = ["input : required distance [m] = 50.0", "segments [-] = 0",
                 "archer alt [m] = -50.00", "wind [m/s] = 0.00",
                 "output: launch angle [rad] = 0.8103", "crossing [-] = 1 (rising crossing)",
                 "hit_speed [m/s] = 91.394", "hit_energy [J] = 94.720", "hit_time [s] = 0.727",
                 "hit_angle [rad] = 0.7584", "hit_x [m] = 49.999", "hit_y [m] = 0.000",
                 "apex [m] = 0.000"]
        at = -1
        for line in order:
            self.assertIn(line, text, f"missing from the transcript: {line!r}\n{text}")
            now = text.index(line)
            self.assertGreater(now, at, f"{line!r} is out of order:\n{text}")
            at = now
        self.assertNotIn("final_h", text, "the highest point is apex; the old label must not reappear")

    def test_the_answer_block_aligns_its_equal_signs(self) -> None:
        '''The answer is a table: one column for the labels, one for the '=' - both interfaces.'''
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _typed(["d=50 l=-50", "q"]):
            arrow.Arrow().interactive()
        repl: list[str] = _block(out.getvalue())
        self.assertGreaterEqual(len(repl), 13, f"the answer block is not there:\n{out.getvalue()}")
        import sys
        cli: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(cli), _argv([sys.argv[0], "d=50"]):
            arrow.main()
        for name, rows in (("the REPL", repl), ("the command line", _block(cli.getvalue()))):
            with self.subTest(interface=name):
                self.assertGreaterEqual(len(rows), 8, f"{name} prints no block: {rows}")
                columns: set[int] = {row.index("=") for row in rows}
                self.assertEqual(len(columns), 1,
                                 f"{name}: the '=' drifts across columns {sorted(columns)}:\n"
                                 + "\n".join(rows))
                self.assertEqual({row[7] for row in rows}, {" "},
                                 f"{name}: labels must start in one column:\n" + "\n".join(rows))
                self.assertNotIn(" ", {row[8] for row in rows},
                                 f"{name}: a label is padded away from the column:\n"
                                 + "\n".join(rows))

    def test_repl_and_cli_print_the_help(self) -> None:
        '''`h` in the REPL and -h / --help on the command line: one screen, no shot, no crash.'''
        body: str = arrow.help_text(arrow.Arrow()).strip()
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _typed(["h", "q"]):
            arrow.Arrow().interactive()
        text: str = out.getvalue()
        self.assertIn(body, text, "the REPL's h did not print the help screen")
        for claim in ("usage:", "s=0|1|2", "0 auto - the faster route", "the plunging shot",
                      "falls back to the other", "the = is optional", "apex the top of the arc",
                      "arrow.py s=2 d=120 l=-50", "terminal speed", "flat target line",
                      "3 lofted", "steepest_angle", "arrow.py s=3 d=60",
                      "hit_x, hit_y where it ended", "hit_speed, hit_energy how it arrives",
                      "crossing which one answered"):
            self.assertIn(claim, body, f"the help does not say {claim!r}: {body}")
        for flag in ("-h", "--help"):
            got: io.StringIO = io.StringIO()
            with contextlib.redirect_stdout(got), _argv([sys.argv[0], flag]):
                arrow.main()
            with self.subTest(flag=flag):
                self.assertEqual(got.getvalue().strip(), body,
                                 "-h and the REPL's h must print one screen, not two")

    def test_the_repl_refuses_nonsense_without_dying(self) -> None:
        '''A bad token says which token, and the session carries on to answer the next line.'''
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _typed(["x=1", "s=7", "d=50", "q"]):
            arrow.Arrow().interactive()
        text: str = out.getvalue()
        self.assertIn("unknown argument 'x=1'", text)
        self.assertIn("segments must be 0 (auto), 1 (rising), 2 (falling) or 3 (lofted); got 7.0",
                      text)
        self.assertIn("output: launch angle [rad] =", _flat(text),
                      "the session must survive a bad line")
        self.assertEqual(arrow.Arrow().segments, 0, "a refused s= must not have been applied")

    def test_apex_is_the_highest_point_and_there_is_only_one_of_it(self) -> None:
        '''One name for the highest point, and no second name that could hold a second truth.'''
        arw = arrow.Arrow()
        arw.archer_alt, arw.segments = -50., 2
        arw.get_angle(120.)
        self.assertGreater(arw.apex, 300., "the plunging shot's apex is well above the sight line")
        self.assertGreater(arw.apex, arw.hit_y, "the apex is above the point the shot ended at")
        self.assertFalse(hasattr(arw, "final_h"),
                         "the old name is gone: this is the first published version, so there is "
                         "nothing to stay compatible with")

    def test_unreachable_names_both_crossings(self) -> None:
        '''The refusal must say what was tried, not just that something failed.'''
        arw = arrow.Arrow()
        arw.max_dist = 700.                            # past the falling branch's own ceiling
        with self.assertRaises(arrow.Unreachable) as got:
            arw.get_angle(650.)
        msg: str = str(got.exception)
        self.assertIn("rising", msg)
        self.assertIn("falling", msg)
        self.assertIn("650.0", msg)

    # ------------------------------------------------------------------- cli -----
    def test_cli_settings_come_before_the_question(self) -> None:
        '''Regression: `d=50 l=10 s=2` used to answer the level shot, ignoring l= and s=.'''
        import sys
        for order in (["d=50", "l=10", "s=2", "w=3"], ["l=10", "s=2", "w=3", "d=50"]):
            out: io.StringIO = io.StringIO()
            argv: list[str] = sys.argv[:1] + order                # type: ignore[assignment]
            with contextlib.redirect_stdout(out), _argv(argv):
                arrow.main()
            text: str = out.getvalue()
            with self.subTest(args=order):
                self.assertIn("crossing [-] = 2 (falling crossing)", _flat(text))
                self.assertIn("-0.17356", text, f"the elevated shot, whatever the order: {text}")

    def test_cli_reports_the_crossing_used(self) -> None:
        '''crossing must be visible where the user reads the answer.'''
        import sys
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _argv([sys.argv[0], "d=3"]):
            arrow.main()
        self.assertIn("crossing [-] = 1 (rising crossing)", _flat(out.getvalue()))
        self.assertNotIn("note:", out.getvalue(), "auto asked for nothing, so the CLI says nothing")

    def test_cli_prints_the_note_where_the_number_is_read(self) -> None:
        '''A substitution the user did not ask for has to appear beside the angle, not in a field.'''
        import sys
        out: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(out), _argv([sys.argv[0], "s=2", "d=3"]):
            arrow.main()
        text: str = out.getvalue()
        self.assertIn("crossing [-] = 1 (rising crossing)", _flat(text))
        self.assertIn("note: you asked for the falling crossing", text)


@contextlib.contextmanager
def _typed(lines: t.Iterable[str]) -> t.Iterator[None]:
    '''builtins.input replayed from `lines`, then end-of-input - which quits the REPL.'''
    import builtins
    queue: list[str] = list(lines)

    def fake(prompt: str = "") -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    keep = builtins.input
    builtins.input = fake
    try:
        yield
    finally:
        builtins.input = keep


@contextlib.contextmanager
def _argv(argv: list[str]) -> t.Iterator[None]:
    '''sys.argv patched for one main() call.'''
    import sys
    keep: list[str] = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = keep


def main() -> None:
    """Do unit tests."""
    unittest.main()


if __name__ == "__main__":
    main()
