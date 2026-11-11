#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) Jan Hercek
"""
Closed-form (loop-less) trajectory of an arrow launched from a (cross)bow.  Default values fit
the MAG340.  Self-contained: stdlib only, nothing outside this file is needed to import, run or
trust it.  The public surface is Arrow, Unreachable, the four questions get_angle, steepest_angle,
get_horizontal_distance and get_vertical_distance, and the read-back they fill (hit_*, plus
crossing, apex and note); the inside is closed
form, with no time step anywhere.  segments defaults to 0 = auto: get_angle asks the rising
crossing first and the falling one second, and crossing says which of them answered.  Asking for
one crossing (s=1 rising, s=2 falling) is never a dead end: when that crossing cannot reach, the
answer comes from the other one and note says so.  steepest_angle, or s=3, answers the other
route to the same target - the lofted one, the high shot over a wall - on any target, including one
level with the bow.

Both interfaces speak one grammar - a= d= s= l= w= in any order, the = optional - and `arrow.py -h`
or the REPL's `h` print the help.  The highest point of the arc is apex: it is not a quantity
at the end of the flight, and a second name for one arc would be two truths about it.

=============================================================================
THE PHYSICS - read this before changing anything below
=============================================================================
Point mass, constant g, quadratic drag always opposite the velocity:

    d vx/dt = -k |v| vx        d vy/dt = -g - k |v| vy        k = 0.5 rho Cd A / m  [1/m]

1) THE FIRST INTEGRAL (the 19th-century hodograph of the quadratic-drag problem).
   Let theta be the flight-path angle atan2(vy, vx).  Because drag is parallel to the
   velocity, dv/dt and dtheta/dt combine into a LINEAR equation for u_ = 1/v**2:

        du_/dtheta + 2 tan(theta) u_ = -(2k/g) sec(theta)

   whose exact solution, with s = sin(theta) and G(s) = s/(1-s**2) + atanh(s), is

        1/v**2 = cos**2(theta) * [ 1/(v0**2 cos**2(alpha)) + (k/g)(G(sin alpha) - G(sin theta)) ]

   Two limits check the whole thing for free: k -> 0 gives v = v0 cos(alpha)/cos(theta)
   (horizontal speed conserved) and theta -> -pi/2 gives v = sqrt(g/k), terminal speed.

2) THE COORDINATE.  Position and time are quadratures over the flight path, but theta is a
   dreadful variable near a vertical plunge: a shot that ends falling straight down spends
   its last hundred metres inside a sliver of theta of width 1e-5, so nothing about it can
   be resolved - by a step or by a node.  The variable that straightens it out is

        u = asinh(tan(theta))     <=>  tan(theta) = sinh(u),  cos(theta) = 1/cosh(u),
                                      sin(theta) = tanh(u),    G(sin theta) = sinh(u)cosh(u)+u

   everything below is then smooth, explicit and free of trig except sinh/cosh of moderate
   arguments, and the pole theta = -pi/2 moves off to u = -infinity:

        P(u) = 1 / (c - (k/g)(sinh(u) cosh(u) + u)),   v**2 = P / cos**2(theta) = P cosh**2(u)

        x = (1/g) Int P cosh(u) du        y = y0 + (1/g) Int P sinh(u) cosh(u) du
        t = (1/g) Int sqrt(P) cosh(u) du

   with the analytic derivatives Newton uses:  dx/du = -P cosh(u)/g,  dy/du = -P sinh(u)
   cosh(u)/g.  Setting k = 0 collapses these to the textbook vacuum formulas term by term
   (reproduced to nanometres), and every flight agrees with an independent RK4 integration
   of the same equations to under 0.01 mm at 612 m, including the near-vertical shots.

3) NO LOOPS.  The three quadratures use a fixed Gauss-Legendre rule on panels of at most
   PANEL_U in u: a constant number of node evaluations, no error accumulation, no stability
   limit, no step size to choose.  Solving "at which u does the arrow reach x / cross the
   sight line" and "which launch angle gives distance d" is root finding - a monotone
   transcendental equation has no elementary inverse - but every such search is bounded:
   a bracket, at most BISECT_STEPS bisections, at most NEWTON_STEPS Newton steps, at most
   MAX_MARCH probes.  Nothing in this file integrates through time.

4) WIND is exact, not fitted: a steady wind is a Galilean shift.  The trajectory is solved
   in the air frame from the launch velocity (v0 cos a - wind, v0 sin a) and the ground
   position is x_air + wind * t.  (Verified against an RK4 that computes drag from the
   relative velocity: they agree to 0.00 mm.)
=============================================================================
"""
import math
import typing as t

VERSION = "3.1.0"


class Unreachable(ValueError):
    """Target cannot be hit: check segments, archer_alt, angle, distance, wind."""


class BadInput(Unreachable):
    """A parameter is outside what the model can represent (a setup error, not a shot).

    A subclass of Unreachable on purpose: code that already catches Unreachable keeps
    working unchanged, and code that wants to tell "that target cannot be hit" from "that
    is not a question the model can answer" still can.
    """


class _Miss(Unreachable):
    """One crossing's refusal, carrying how close it got.  Internal: get_angle re-raises it.

    Its str() is the reason, and it never says "closest -1.0 m": -1. is the search's own
    marker for "no angle on this branch ever gets back to the sight line", which is a fact
    about the branch, not a distance, and must not read like one.
    """

    def __init__(self, seg: int, best: float, best_angle: float) -> None:
        which: str = _CROSSING[seg]
        why: str = (f"the {which} crossing never gets back to the sight line at any angle"
                    if best < 0. else
                    f"the {which} crossing got no closer to it than {best:.1f} m "
                    f"(at {best_angle:.4f} rad)")
        super().__init__(why)
        self.seg: int = seg
        self.best: float = best
        self.best_angle: float = best_angle


# --- the fixed Gauss-Legendre rule on [-1,1] and every loop bound in the file ---------
_NODES: t.Final[tuple[float, ...]] = (
    -0.96028985649753627, -0.79666647741362678, -0.52553240991632904, -0.18343464249564978,
    0.18343464249564978, 0.52553240991632904, 0.79666647741362678, 0.96028985649753627)
_WEIGHTS: t.Final[tuple[float, ...]] = (
    0.10122853629037706, 0.22238103445337443, 0.31370664587788688, 0.36268378337836166,
    0.36268378337836166, 0.31370664587788688, 0.22238103445337443, 0.10122853629037706)

PI_2: t.Final[float] = math.pi / 2.0
U_LIMIT: t.Final[float] = 22.0          # |u|: |tan(theta)| ~ 2e9, far past any real arrow
PANEL_U: t.Final[float] = 0.70          # rad-ish, max width of one quadrature panel in u
MAX_PANELS: t.Final[int] = 64           # bound on quadrature panels
U_STEP: t.Final[float] = 1.00           # |tan(theta)| grows x e per step of the bracket walk
BISECT_STEPS: t.Final[int] = 16         # bound on bisection; Newton does the last mile
NEWTON_STEPS: t.Final[int] = 5          # bound on Newton polish (it is quadratic: 5 is plenty)
ANGLE_MARCH: t.Final[float] = 0.05      # growth of one bracketing probe in get_angle
ANGLE_FLOOR: t.Final[float] = 0.01      # rad, smallest bracketing probe (a shot is ~0.007)
MAX_MARCH: t.Final[int] = 40            # bound on the launch-angle probes
ANGLE_STEPS: t.Final[int] = 48          # bound on one branch refinement
ANGLE_EPS: t.Final[float] = 1e-12       # rad, bracket collapse
_CROSSING: t.Final[tuple[str, ...]] = ("", "rising", "falling", "lofted")   # indexed by segments
FAR_STAMP: t.Final[tuple[str, ...]] = (            # the settings the far limit is remembered
    "gravity_accel", "air_dens", "drag_coef", "ar_mass", "ar_launch_velo", "ar_dia",     # against
    "ar_vane_n", "ar_vane_height", "ar_vane_thick", "wind", "archer_alt", "ar_launch_alt")


# --- the REPL and the command line speak one grammar ------------------------------------
KEYS: t.Final[tuple[str, ...]] = ("a", "d", "s", "l", "w")
HELP_KEYS: t.Final[tuple[str, ...]] = ("h", "help", "-h", "--help", "?")
QUIT_KEYS: t.Final[tuple[str, ...]] = ("q", "quit", "exit")


def parse_keys(tokens: t.Iterable[str]) -> dict[str, float]:
    """`a=0.05` or `a0.05` -> {'a': 0.05}: one grammar for both interfaces, = optional.

    The REPL used to insist there were no `=` signs and the command line to insist there were,
    and typing the wrong form got you `could not convert string to float: '=50'` - a syntax
    error dressed up as arithmetic. Both spellings are accepted by both interfaces now, and a
    token neither grammar can read names itself instead of failing inside float().
    """
    nums: dict[str, float] = {}
    for item in tokens:
        key, sep, val = item.partition("=")
        if not sep:                                   # the REPL's older, tighter spelling
            key, val = item[:1], item[1:]
        if key not in KEYS:
            raise BadInput(f"unknown argument {item!r} (known: {' '.join(KEYS)}=; h prints the help)")
        try:
            nums[key] = float(val)
        except ValueError:
            raise BadInput(f"bad value in argument {item!r} (want a number after {key}=)") from None
    return nums


def apply_settings(arw: "Arrow", nums: dict[str, float]) -> None:
    """Apply the settings (s=, l=, w=) and ask nothing: they are applied before the question.

    Split out of answering because order used to matter. `main()` walked the arguments once, left
    to right, so its own documented example `d=50 l=10 s=2 w=3` answered the level shot and
    silently ignored l=, s= and w=. A line of keys is a state, not a script.
    """
    if "s" in nums:
        if nums["s"] not in (0., 1., 2., 3.):
            raise BadInput(f"segments must be 0 (auto), 1 (rising), 2 (falling) or 3 (lofted); "
                           f"got {nums['s']}")
        arw.segments = int(nums["s"])
    if "l" in nums:
        arw.archer_alt = nums["l"]
    if "w" in nums:
        arw.wind = nums["w"]


def help_text(arw: "Arrow") -> str:
    """The whole interface on one screen: what `arrow.py -h` and the REPL's `h` print."""
    return "\n".join([
        f"arrow.py({VERSION}) - the trajectory of an arrow from a (cross)bow, solved in closed",
        "form: no time step, no marching over flight time. Defaults fit a MAG340.",
        "",
        "usage:",
        "  arrow.py                                     the REPL, one command per line",
        "  arrow.py [a=rad | d=m] [s=0|1|2|3] [l=m] [w=m/s]      one shot, then exit",
        "  arrow.py -h | --help                         this text",
        "",
        "keys - the = is optional, and the order does not matter:",
        "  d=m          which angle hits this ground distance?    -> launch angle",
        "  a=rad        where does this launch angle land?        -> flight distance",
        "  a=rad d=m    how high is the arrow at that x, on that shot?",
        "  s=0|1|2|3    which shot you want, out of the ones that hit this target:",
        "                 0 auto - the faster route: rising if it can reach, else falling",
        "                 1 rising  - the arrow is still climbing at the target",
        "                 2 falling - the plunging shot, the one that clears a wall",
        "               a crossing that cannot reach falls back to the other and says so",
        "               in the note; Unreachable comes only when neither can hit",
        "                 3 lofted  - the steep twin of the flat shot: the high one over an",
        "                       obstacle, on any target, level ones included (steepest_angle)",
        "  l=m          archer's altitude relative to the target (+ = archer above)",
        "  w=m/s        steady wind along the line of fire (+ = tailwind)",
        "",
        "read back after a shot - the numbers of the shot just given, filled by whichever call",
        "returned it:  hit_x, hit_y where it ended   hit_speed, hit_energy how it arrives,",
        "              hit_time how long, hit_angle at what angle, crossing which one answered",
        "              (1 rising, 2 falling), apex the top of the arc, note what the solver said",
        "",
        "examples:",
        "  arrow.py d=50                      level 50 m shot",
        "  arrow.py d=50 l=10 s=2 w=3         same target 10 m below, plunging, 3 m/s tailwind",
        "  arrow.py s=2 d=120 l=-50           the shot over a wall: apex 311 m, 84 degrees",
        "  arrow.py s=3 d=60                  the high shot to a target level with the bow",
        "  arrow.py a=0.05                    how far 0.05 rad carries",
        "  arrow.py a=0.05 d=60               how high the arrow is at 60 m on that shot",
        "  in the REPL:  d=50 l=-50           several keys per line, settings then question",
        "                h                    this text",
        "                q                    quit",
        "",
        (f"questions are taken from {arw.min_dist:g} m to {arw.max_dist:g} m, the model's own "
         f"maximum (set max_dist for a smaller limit); angle open at +-pi/2; "
         f"terminal speed {arw.terminal_speed():.1f} m/s."),
        "The model is a point mass with quadratic drag, steady wind, flat target line: see the",
        "header of this file for the derivation and README.md for what it leaves out.",
    ])


INDENT: t.Final[str] = "       "          # seven spaces: the width of "input :" and "output:"
Row: t.TypeAlias = tuple[str, str, str, str]        # prefix, label, unit, value


def answer_block(rows: t.Sequence[Row]) -> str:
    """One answer as a table: prefix, label [unit], then the value under a single aligned '='.

    Each row is (prefix, label, unit, value); the prefix is "input :", "output:" or INDENT for the
    lines that continue the same question, so all three prefixes are seven characters wide and the
    labels start in one column. The `=` used to sit wherever each label happened to end, so
    `[m/s]=` and `[-]=` never lined up and the block read as nine unrelated sentences.
    """
    width: int = max(len(f"{label} [{unit}]") for _, label, unit, _ in rows)
    return "\n".join(f"{prefix} {label} [{unit}]".ljust(8 + width) + f"= {value}"
                     for prefix, label, unit, value in rows)


def settings_rows(arw: "Arrow", rows: list[Row]) -> None:
    """What the arrow is set to. The settings are part of the question, so they go with it."""
    rows.append((INDENT, "segments", "-", str(arw.segments)))
    rows.append((INDENT, "archer alt", "m", f"{arw.archer_alt:.2f}"))
    rows.append((INDENT, "wind", "m/s", f"{arw.wind:.2f}"))


def shot_rows(arw: "Arrow", rows: list[Row]) -> None:
    """The numbers of the shot just returned, crossing first: it says which shot the rest describe."""
    shot: str = f"{arw.crossing} ({_CROSSING[arw.crossing]} crossing)"
    rows.append((INDENT, "crossing", "-", shot))
    rows.append((INDENT, "hit_speed", "m/s", f"{arw.hit_speed:.3f}"))
    rows.append((INDENT, "hit_energy", "J", f"{arw.hit_energy:.3f}"))
    rows.append((INDENT, "hit_time", "s", f"{arw.hit_time:.3f}"))
    rows.append((INDENT, "hit_angle", "rad", f"{arw.hit_angle:.4f}"))
    rows.append((INDENT, "hit_x", "m", f"{arw.hit_x:.3f}"))
    rows.append((INDENT, "hit_y", "m", f"{arw.hit_y:.3f}"))
    rows.append((INDENT, "apex", "m", f"{arw.apex:.3f}"))


def _quad(f: t.Callable[[float], float], u_lo: float, u_hi: float) -> float:
    """Integral of f over [u_lo, u_hi] with a fixed count of Gauss-Legendre nodes.

    Uniform panels in u of at most PANEL_U: in this coordinate the integrands are smooth
    everywhere, including for a shot that ends in a vertical plunge (u -> -inf), where the
    naive theta form blows up like 1/cos(theta).
    """
    if u_hi <= u_lo:
        return 0.0
    n_pan: int = min(MAX_PANELS, max(1, int((u_hi - u_lo) / PANEL_U) + 1))
    width: float = (u_hi - u_lo) / n_pan
    half: float = 0.5 * width
    acc: float = 0.0
    for p in range(n_pan):                     # bounded: n_pan <= MAX_PANELS
        mid: float = u_lo + width * (p + 0.5)
        acc += half * sum(wt * f(mid + half * xi)
                          for xi, wt in zip(_NODES, _WEIGHTS, strict=True))
    return acc


def _newton(f: t.Callable[[float], float], df: t.Callable[[float], float],
            lo: float, hi: float, x: float) -> float:
    """Newton from x, never leaving [lo,hi]; fixed step count."""
    for _ in range(NEWTON_STEPS):
        d: float = df(x)
        if d == 0.0:
            break
        nxt: float = x - f(x) / d
        if not lo < nxt < hi or abs(nxt - x) < 1e-14 * max(1.0, abs(x)):
            break
        x = nxt
    return x


def _root(f: t.Callable[[float], float], lo: float, hi: float,
          df: t.Callable[[float], float] | None = None) -> float | None:
    """Root of a monotone f on [lo,hi] - f may increase or decrease; None if not bracketed."""
    fa: float = f(lo)
    fb: float = f(hi)
    if fa == 0.0:
        return lo
    if fb == 0.0:
        return hi
    if fa * fb > 0.0:
        return None
    a: float = lo
    b: float = hi
    for _ in range(BISECT_STEPS):
        m: float = 0.5 * (a + b)
        fm: float = f(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if b - a < 1e-13 * max(1.0, abs(b)):
            break
    x: float = 0.5 * (a + b)
    return _newton(f, df, lo, hi, x) if df is not None else x


class _Flight:
    """One trajectory, fully described by its air-frame launch state.

    Everything is a function of u = asinh(tan(theta)); the theta-facing wrappers convert.
    """

    __slots__ = ("_c", "_kg", "alpha", "g", "u_alpha", "v0", "y0")

    def __init__(self, v0: float, alpha: float, k: float, g: float, y0: float) -> None:
        self.v0, self.alpha, self.g, self.y0 = v0, alpha, g, y0
        self._kg: float = k / g
        self.u_alpha: float = math.asinh(math.tan(alpha))
        # c = 1/(v0^2 cos^2 alpha) + (k/g) G(sin alpha), with G(sin u) = sinh u cosh u + u
        ch: float = math.cosh(self.u_alpha)
        self._c: float = ch * ch / (v0 * v0) + self._kg * (
            math.sinh(self.u_alpha) * ch + self.u_alpha)

    # ---------------------------------------------------------------- the core --
    def p(self, u: float) -> float:
        """P(u) = v**2 cos**2(theta): the smooth core of the first integral."""
        return 1.0 / (self._c - self._kg * (math.sinh(u) * math.cosh(u) + u))

    def v2_u(self, u: float) -> float:
        ch: float = math.cosh(u)
        return self.p(u) * ch * ch

    def v_u(self, u: float) -> float:
        return math.sqrt(max(self.v2_u(u), 0.0))

    # One quadrature each - a root search calls only the one it needs, never all three.
    def x_u(self, u: float) -> float:
        """Air-frame x when the flight-path angle has fallen to u (u <= u_alpha)."""
        return _quad(lambda q: self.p(q) * math.cosh(q), u, self.u_alpha) / self.g

    def y_u(self, u: float) -> float:
        """Height above the target line at flight-path angle u."""
        return self.y0 + _quad(lambda q: self.p(q) * math.sinh(q) * math.cosh(q),
                               u, self.u_alpha) / self.g

    def t_u(self, u: float) -> float:
        """Flying time spent getting to flight-path angle u."""
        return _quad(lambda q: math.sqrt(max(self.p(q), 0.0)) * math.cosh(q),
                     u, self.u_alpha) / self.g

    # ------------------------------------------------------- theta-facing views --
    @staticmethod
    def u_of(th: float) -> float:
        return math.asinh(math.tan(th))

    @staticmethod
    def theta_of(u: float) -> float:
        return math.atan(math.sinh(u))

    def v2(self, th: float) -> float:
        return self.v2_u(self.u_of(th))

    def v(self, th: float) -> float:
        return self.v_u(self.u_of(th))

    def x(self, th: float) -> float:
        return self.x_u(self.u_of(th))

    def y(self, th: float) -> float:
        return self.y_u(self.u_of(th))

    def t(self, th: float) -> float:
        return self.t_u(self.u_of(th))


class Arrow:
    """Calculates an arrow trajectory. Outputs angle, horizontal and/or vertical distance,..."""

    DIST_ACCURACY: t.Final[float] = 0.05   # m, how close a returned angle must really land

    def __init__(self) -> None:
        """Attributes below are based on standard configuration of the MAG340 crossbow."""
        # input:
        self.archer_alt: float = 0.  # m, archer altitude relative to the target
        self.segments: t.Literal[0, 1, 2, 3] = 0  # 0: auto - ask 1, then 2 (the default)
        #  1: first crossing of the sight line (rising),
        #  2: the far crossing (falling) - the normal shot
        #  3: the lofted route - the steepest angle that hits, on either crossing
        # mag340 arrows:
        self.ar_launch_velo: float = 340. / 3.2808398950131  # fps to m/s, arrow launch velocity
        self.ar_mass: float = 350. * 6.479891e-5  # grn to kg, arrow mass
        self.ar_dia: float = 0.346 * 0.0254       # inch to m, arrow diameter
        self.ar_vane_height: float = 0.015        # m, arrow vane height
        self.ar_vane_thick: float = 0.0008        # m, arrow vane thickness
        self.ar_vane_n: int = 3                   # -, arrow vanes
        self.ar_launch_alt: float = -0.025        # m, distance from eye to arrow at launch
        self.min_dist: float = 0.01               # m, minimal distance to target
        # max_dist is a property: derived from the model unless a number is assigned to it.
        self._far: float | None = None            # m, an imposed limit, or None to derive one
        self._far_cache: float | None = None      # m, the derived limit, remembered
        self._far_key: tuple[float, ...] | None = None   # settings that limit was derived at
        # aerodynamics:
        self.drag_coef: float = 0.4        # -, drag coefficient: shape+friction+induction
        self.air_dens: float = 1.225       # kg/m^3, air density
        self.gravity_accel: float = 9.81   # m/s^2, gravity acceleration
        self.wind: float = 0.              # m/s, + = tailwind (air moves toward the target)
        # output:
        self.hit_angle: float = 0.   # rad, arrow final angle (ground frame)
        self.hit_speed: float = 0.   # m/s, arrow final velocity (ground frame)
        self.hit_energy: float = 0.   # J, kinetic energy at impact
        self.hit_time: float = 0.   # s, arrow flying time
        self.hit_x: float = 0.   # m, horizontal distance
        self.hit_y: float = 0.   # m, vertical distance
        self.apex: float = 0.     # m, highest point of the arc (not a quantity at the end)
        self.crossing: int = 0      # -, which crossing answered the last question
        self.note: str = ""  # -, what the shot asked for said about its own choice

    # ------------------------------------------------------------------ model --
    def drag_k(self) -> float:
        """Drag constant k = 0.5 rho Cd A / m in 1/m;  k = 0 would be a vacuum."""
        area: float = (math.pi * (self.ar_dia / 2.) ** 2
                       + self.ar_vane_n * self.ar_vane_height * self.ar_vane_thick)
        return 0.5 * self.air_dens * self.drag_coef * area / self.ar_mass

    def terminal_speed(self) -> float:
        """sqrt(g/k): the speed this arrow falls at - the model's own drag yardstick."""
        k: float = self.drag_k()
        return math.sqrt(self.gravity_accel / k) if k > 0. else float("inf")

    def _beyond(self, where: str, distance: float) -> Unreachable:
        """The refusal for a distance the interface takes no question about.

        It used to print `target_distance=700.0`, which told an archer nothing about what to ask
        instead.  The limit is the model's own far limit now, so the message names both numbers a
        reader could want: where asking stops, and how far this arrow can carry at all - the same
        figure unless max_dist has been set by hand.
        """
        return Unreachable(
            f"in {where}: {distance:g} m is past the limit {self.max_dist:.1f} m "
            f"(this arrow carries at most {self._far_range():.1f} m at these settings; the nearest "
            f"question the model takes is {self.min_dist:g} m)")

    def _check(self) -> None:
        if self.segments not in (0, 1, 2, 3):
            raise BadInput("segments must be 0 (auto), 1 (rising), 2 (falling) or 3 (lofted), "
                           f"got {self.segments!r}")
        if self.ar_mass <= 0. or self.gravity_accel <= 0. or self.ar_launch_velo <= 0.:
            raise BadInput("ar_mass, gravity_accel and ar_launch_velo must be positive")
        if self.drag_coef < 0. or self.air_dens < 0. or self.ar_dia < 0.:
            raise BadInput("drag_coef, air_dens and ar_dia must not be negative")
        if self.wind >= self.ar_launch_velo:
            # Not a missed shot but an impossible bow: the air-frame forward speed is
            # v0 cos(angle) - wind, which is never positive once the wind is at or beyond v0, so no
            # angle exists to search.  Saying so here keeps it a setup error instead of letting the
            # per-angle guard below report it as a crossing that cannot reach.
            raise BadInput(f"wind {self.wind} m/s is at or beyond the launch speed "
                           f"{self.ar_launch_velo:.6g} m/s: no angle gets the arrow forward")

    def _launch(self, angle: float) -> _Flight:
        """Air-frame trajectory for a *ground* launch angle; wind is a Galilean shift."""
        self._check()
        if not -PI_2 < angle < PI_2:
            raise BadInput(f"launch angle out of range: {angle=}")
        if abs(angle) > PI_2 - 1e-4:
            raise BadInput(f"launch angle too close to vertical: {angle=}")
        vx: float = self.ar_launch_velo * math.cos(angle) - self.wind
        vy: float = self.ar_launch_velo * math.sin(angle)
        if vx <= 1e-9:
            raise BadInput(f"headwind {self.wind} m/s beats the arrow at angle {angle:.5f} rad")
        return _Flight(math.hypot(vx, vy), math.atan2(vy, vx),
                       self.drag_k(), self.gravity_accel, self.archer_alt + self.ar_launch_alt)

    # ----------------------------------------------------------- u solving ------
    def _descend(self, f: t.Callable[[float], float], hi: float) -> tuple[float, float] | None:
        """Walk u downward from hi until f changes sign; return the tight bracket [lo, hi].

        Brackets are grown, not assumed: the far end is only pushed toward -U_LIMIT as far
        as the trajectory actually goes.  A level shot crosses the sight line near u = -1 and
        costs two probes; a shot that ends in a vertical plunge pays for the depth it needs.
        """
        b: float = hi
        fb: float = f(b)
        for _ in range(MAX_PANELS):                  # bounded walk downward
            a: float = max(b - U_STEP, -U_LIMIT)
            fa: float = f(a)
            if fa * fb <= 0.0:
                return a, b
            if a <= -U_LIMIT:
                return None
            b, fb = a, fa
        return None

    def _x_of(self, fl: _Flight, u: float) -> float:
        """Ground-frame x at flight-path angle u: wind drifts the arrow for the whole flight."""
        x: float = fl.x_u(u)
        return x + self.wind * fl.t_u(u) if self.wind else x

    def _y_of(self, fl: _Flight, u: float) -> float:
        return fl.y_u(u)

    def _u_at_x(self, fl: _Flight, x_req: float) -> float | None:
        """Flight-path angle (u) of the first moment ground-x == x_req."""
        def f(u: float) -> float:
            return self._x_of(fl, u) - x_req

        def df(u: float) -> float:
            ch: float = math.cosh(u)
            d: float = -fl.p(u) * ch / fl.g
            if self.wind:
                d -= self.wind * math.sqrt(max(fl.p(u), 0.0)) / fl.g
            return d

        span = self._descend(f, fl.u_alpha)
        return None if span is None else _root(f, span[0], span[1], df)

    def _u_at_line(self, fl: _Flight) -> float | None:
        """Flight-path angle (u) at which the arrow reaches the sight line y = 0, or None."""
        def f(u: float) -> float:
            return self._y_of(fl, u)

        def df(u: float) -> float:
            return -fl.p(u) * math.sinh(u) * math.cosh(u) / fl.g

        if self.segments == 1:                       # the rising crossing: u in [0, u_alpha]
            if fl.u_alpha <= 0.:
                return None
            return _root(f, 0., fl.u_alpha, df)
        hi: float = min(fl.u_alpha, 0.)              # the falling crossing: u in (-inf, that]
        if self._y_of(fl, hi) <= 0.:                 # never gets above the sight line
            return None
        span = self._descend(f, hi)
        return None if span is None else _root(f, span[0], span[1], df)

    # ------------------------------------------------------------- reporting ----
    def _report(self, fl: _Flight, u: float, x: float, y: float, tt: float, v: float) -> None:
        """Fill the read-back of the shot, in the ground frame (undo the Galilean shift)."""
        th: float = _Flight.theta_of(u)
        vy: float = v * math.sin(th)
        vx: float = v * math.cos(th) + self.wind
        u_apex: float = min(max(0., u), fl.u_alpha)  # top of the *flown* arc
        self.hit_angle = math.atan2(vy, vx)
        self.hit_speed = math.hypot(vx, vy)
        self.hit_energy = 0.5 * self.ar_mass * self.hit_speed ** 2
        self.hit_time = tt
        self.hit_x = x
        self.hit_y = y
        self.apex = max(y, self._y_of(fl, u_apex))

    def _range_of(self, angle: float) -> float:
        """Distance at which the arrow meets the sight line.  Cheap probe: fills nothing.
        -1. means 'this angle never gets to the sight line' - a search answer, not a shot."""
        fl: _Flight = self._launch(angle)
        u: float | None = self._u_at_line(fl)
        return -1. if u is None else self._x_of(fl, u)

    # --------------------------------------------------------------- public -----
    def get_horizontal_distance(self, launch_angle: float) -> float:
        """Input = radians, Output = meters (where the arrow meets the sight line).

        With segments = 0 (auto) the forward questions use the crossing get_angle last reported in
        crossing - the same trajectory you were just given - and the falling one, which is the range
        of the shot, when no shot has been asked for yet.  Set segments = 1 to ask for the rising
        crossing whatever happened before.  crossing says which one answered; note is
        get_angle's to write and is left alone here, so reading it after a verification still
        tells you what get_angle thought about its own choice.
        """
        keep: int = self.segments
        self.segments = (self.crossing if self.crossing in (1, 2) else 2) if keep == 0 else keep
        try:
            fl: _Flight = self._launch(launch_angle)
            u: float | None = self._u_at_line(fl)
            if u is None:
                raise Unreachable(f"in get_horizontal_distance: at {launch_angle} rad with "
                                  f"segments={self.segments} the arrow never meets the sight line")
            self._report(fl, u, self._x_of(fl, u), self._y_of(fl, u), fl.t_u(u), fl.v_u(u))
            used: int = self.segments
        finally:
            self.segments = keep
        self.crossing = used
        return self.hit_x

    def get_vertical_distance(self, launch_angle: float, horizontal_distance: float) -> float:
        """Input = radians and meters, Output = meters (signed vertical error at that x).

        The crossing follows crossing exactly as in get_horizontal_distance, so
        get_vertical_distance(get_angle(d), x_wall) reads the height of the shot get_angle just
        reported over the wall at x_wall - which is the whole of what clearing a wall asks for.
        """
        self._check()                                 # a broken bow is a setup error, first
        if not self.min_dist <= horizontal_distance <= self.max_dist:
            raise self._beyond("get_vertical_distance", horizontal_distance)
        keep: int = self.segments
        self.segments = (self.crossing if self.crossing in (1, 2) else 2) if keep == 0 else keep
        try:
            fl: _Flight = self._launch(launch_angle)
            u: float | None = self._u_at_x(fl, horizontal_distance)
            reached: bool = u is not None
            if u is None:
                # The arrow never gets that far: report where its flight ended (a negative
                # height) instead of integrating toward it forever.
                end: float | None = self._u_at_line(fl)
                u = -U_LIMIT if end is None else end
            x: float = self._x_of(fl, u)
            self._report(fl, u, horizontal_distance if reached else x,
                         self._y_of(fl, u), fl.t_u(u), fl.v_u(u))
            used = self.segments
        finally:
            self.segments = keep
        self.crossing = used
        return self.hit_y

    def get_angle(self, target_distance: float) -> float:
        """Input = meters. Output = radians: the angle that hits, or Unreachable if nothing can.

        The two crossings of the sight line are two different arrows at the same target, so
        `segments` says which one is wanted and `crossing` says which one answered:

            0  the fastest route to the target - the rising crossing whenever it can reach, and
               the falling one otherwise.  Measurement, not taste: in every geometry where both
               answer (29 of 29 tried) the rising shot is the flatter and far faster one.
            1  the rising crossing; when it cannot reach, the falling one, with note saying
               so.  This is the long-distance case: the flat shot simply stops short.
            2  the falling crossing - the plunging shot that clears a wall - and when the rising
               route also exists, note names it, because it is the shot you would rather have
               fired if nothing was in the way.
            3  the lofted route: steepest_angle - the *steepest* angle that hits, the high shot over
               an obstacle, for any target including one level with the arrow.

        So a crossing that cannot reach is never an error by itself: `Unreachable` comes only when
        neither crossing can hit, and its message says what each one did. `segments` outside 0..3
        or a setup the air cannot support is `BadInput`, which is a setup error and not a miss.
        """
        self._check()                                 # a broken bow is asked about before its range
        if not self.min_dist <= target_distance <= self.max_dist:
            raise self._beyond("get_angle", target_distance)
        want: int = self.segments
        if want == 3:                                 # a different question of the same model
            return self._lofted_angle(target_distance)
        self.note = ""
        answers: dict[int, tuple[float, float]] = {}      # crossing -> (angle, flight time)
        misses: dict[int, str] = {}
        for seg in (1, 2):
            if not self._crossing_can_reach(seg, target_distance):
                misses[seg] = (f"the {_CROSSING[seg]} crossing cannot get that far")
                continue
            try:
                angle: float = self._solve_angle(target_distance, seg)
            except BadInput:
                raise                                  # a setup error is not a missed shot
            except Unreachable as err:
                misses[seg] = str(err)
                continue
            answers[seg] = (angle, self.hit_time)

        chosen: int | None
        if want == 0:
            chosen = 1 if 1 in answers else (2 if 2 in answers else None)
        elif want in answers:
            chosen = want
        else:
            chosen = 1 if 1 in answers else (2 if 2 in answers else None)
        if chosen is None:
            raise Unreachable(
                f"in get_angle: {target_distance} m cannot be hit at archer_alt "
                f"{self.archer_alt} m: {misses.get(1, 'no angle')}; {misses.get(2, 'no angle')}") from None

        # Whatever answered, the numbers handed back must describe *that* angle: the loop above may
        # well have solved the other crossing last, and a falling shot reported with a rising shot's
        # flight time is worse than no number at all.
        keep: int = self.segments
        self.segments = chosen
        try:
            self.get_horizontal_distance(answers[chosen][0])
        finally:
            self.segments = keep
        self.crossing = chosen
        self.note = self._note(want, chosen, answers, misses)
        return answers[chosen][0]

    def steepest_angle(self, target_distance: float) -> float:
        """Input = meters. Output = radians: the *steepest* angle that hits, or Unreachable.

        Anything that can be hit at all can be hit two ways: a flat shot, and a lofted one that goes
        up over what is in the way and comes down on the target.  `get_angle` answers with the
        flattest.  This answers with the steepest - the shot that clears a wall.

        Same physics, same closed form, no new machinery.  On the far side of the angle of maximum
        range the distance falls monotonically from that maximum down to the near-vertical landing
        point, so one bracketed bisection finds it.  Every probe is a closed-form flight and the
        angle comes back only when the model's own forward call agrees it lands within
        DIST_ACCURACY; nothing is marched through flight time.

        Why `segments` could not reach it.  The lofted twin of a target *level with the arrow* is a
        falling crossing, like the flat one, and the branch search on the falling crossing returns
        the flattest hit - so `segments` = 1 and 2 both answer with the flat shot.  This walks down
        from the vertical end instead.  `segments` = 3 asks for it through `get_angle`; the read-back
        block is filled for the shot returned and crossing still names the crossing that answered.

        What the lofted shot costs is written into `note`, in meters and seconds, beside the
        flat route's angle and time.  At the short end the refusal is honest: below about 0.12 m even
        an arrow fired at the model's vertical limit still lands past the target, so there is no
        lofted solution to be had there.
        """
        self._check()                                 # setup errors, not misses
        if not self.min_dist <= target_distance <= self.max_dist:
            raise self._beyond("steepest_angle", target_distance)
        return self._lofted_angle(target_distance)

    def _lofted_angle(self, target_distance: float) -> float:
        """The search behind steepest_angle, private so that get_angle can reach it too.

        Private rather than folded into steepest_angle because the note needs the *flat* route to
        compare against, and asking get_angle for it would make get_angle and steepest_angle call
        each other - a cycle in the call graph, which the test suite forbids. Auto's own order
        (rising, then falling) is repeated here on the same private solver get_angle uses.
        """
        self.note = None
        keep: int = self.segments
        self.segments = 2                             # the lofted solution is a falling crossing
        try:
            peak: float = self._peak_angle()          # the angle of maximum distance
            guard: float = PI_2 - 1e-4                # the model's own vertical limit
            far: float = self._safe_range(peak)       # how far any angle can carry
            near: float = self._safe_range(guard)     # where a near-vertical arrow comes down
            if far < target_distance:
                best: float = self.max_range()
                if best < 0.:
                    raise Unreachable(f"in steepest_angle: at archer_alt {self.archer_alt} m no "
                                      f"angle of this bow brings an arrow to a target "
                                      f"{target_distance} m out")
                raise Unreachable(f"in steepest_angle: nothing reaches {target_distance} m; the "
                                  f"farthest this bow sends an arrow is {best:.1f} m")
            if near < 0.:
                raise Unreachable(f"in steepest_angle: {target_distance} m is too close for the "
                                  f"lofted shot, and at this altitude no arrow comes back down to "
                                  f"the sight line at all")
            if near > target_distance:
                raise Unreachable(f"in steepest_angle: {target_distance} m is too close for the "
                                  f"lofted shot - even at the vertical limit an arrow lands "
                                  f"{near:.2f} m out")
            angle: float | None = _root(lambda a: self._safe_range(a) - target_distance,
                                        peak, guard)  # bounded: BISECT_STEPS closed-form flights
            if angle is None or abs(self._safe_range(angle) - target_distance) > self.DIST_ACCURACY:
                raise Unreachable(f"in steepest_angle: no lofted solution at {target_distance=} "
                                  f"verified against the forward model")
            # What the archer is giving up, in the units the crossing notes use.  The lofted flight
            # is flown again afterwards so the read-back block describes the shot actually returned.
            flat: float | None = None
            flat_t: float = 0.
            for seg in (1, 2):                        # auto's order, asked privately
                if not self._crossing_can_reach(seg, target_distance):
                    continue
                try:
                    flat = self._solve_angle(target_distance, seg)
                    flat_t = self.hit_time
                except BadInput:
                    raise                             # a setup error is not a missed shot
                except Unreachable:
                    continue                          # this crossing has no flat shot either
                break
            self.get_horizontal_distance(angle)
            self.crossing = 2
            if flat is not None and flat < angle - ANGLE_FLOOR:   # nearer than this, the two roots
                self.note = (                   # have merged at max range: one shot, no note
                    f"this is the lofted shot: {math.degrees(angle):.1f} deg, apex "
                    f"{self.apex:.1f} m; the flat route hits the same target at +{flat:.5f} rad "
                    f"in {flat_t:.3f} s - take that one unless something is in the way")
            return angle
        finally:
            self.segments = keep

    def _note(self, want: int, chosen: int, answers: dict[int, tuple[float, float]],
              misses: dict[int, str]) -> str:
        """One line on what was asked for against what was had; empty when they agree.

        The forced crossings are the reason it exists.  `s=2` is how an arrow is put over a wall,
        and it is also how an arrow is lobbed 365 m into the air at a target 3 m away: the setting
        must not be allowed to hide that.  So the note names the crossing that answered and, when
        the other one exists too, its angle and its flight time - the flatter route is never
        withheld behind a setting - and when the wanted crossing is the one that cannot reach, it
        quotes that crossing's own refusal instead of leaving the substitution unexplained.
        """
        other: int = 2 if chosen == 1 else 1
        if want == 0:
            return ""                      # auto asked for nothing; crossing says what it got
        if want == chosen:
            if other not in answers:
                return ""
            angle, tt = answers[other]
            if chosen == 2:
                return (f"this is the high shot, apex {self.apex:.1f} m; the rising crossing hits "
                        f"the same target flatter and faster, at {angle:+.5f} rad in {tt:.3f} s - "
                        f"take that one unless something is in the way")
            return (f"this is the flat shot; the falling crossing also hits it, at "
                    f"{angle:+.5f} rad in {tt:.3f} s - the one for putting an arrow over a wall")
        return (f"you asked for the {_CROSSING[want]} crossing and it cannot hit this target "
                f"({misses.get(want, 'no angle on that branch')}); this is the "
                f"{_CROSSING[chosen]} crossing's answer")

    def _crossing_can_reach(self, seg: int, distance: float) -> bool:
        """Cheap bound that lets the auto search skip a crossing which cannot win.

        The arrow is released target_alt below the sight line, so for the two crossings of that
        line x1 * x2 = v0**2 * 2 * target_alt / g, hence the rising one can never exceed
        v0 * sqrt(2 target_alt / g) - 7.4 m for the default arrow at level ground, which is
        exactly where the falling crossing takes over.  Drag and a headwind only shorten it; a
        tailwind shifts the ground frame, so whenever the wind is not still the bound is not
        trusted and the crossing is searched anyway.
        """
        target_alt: float = -(self.archer_alt + self.ar_launch_alt)
        if seg == 2 or target_alt <= 0.:
            return seg == 2
        if self.wind != 0.:
            return True
        return distance <= self.ar_launch_velo * math.sqrt(2. * target_alt / self.gravity_accel)

    def _solve_angle(self, target_distance: float, seg: int) -> float:
        """The branch search below, pinned to one crossing, with crossing recording the choice."""
        keep: int = self.segments
        self.segments = seg
        try:
            angle: float = self._angle_on_branch(target_distance)
        finally:
            self.segments = keep
        self.crossing = seg
        return angle

    def _angle_on_branch(self, target_distance: float) -> float:
        """The flattest angle on self.segments' branch that lands within DIST_ACCURACY.

        distance(angle) is not a friendly curve.  Below some angle the arrow never reaches
        the sight line at all; then it grows to a maximum (about 40 deg here) and falls back
        to zero at vertical; and where it appears, it *jumps* - so a solver that simply
        hunts for distance == target can stop on a jump and report a hit its own forward
        model denies.  So this searches two
        monotone branches in turn and *verifies* the winner with the forward model:

          phase 1  the flattest angle whose distance already reaches the target;
          phase 2  if that landed on a jump, the flattest angle above it landing at or short
                   of the target on the way down;
          phase 3  the jump itself (see _jump_angle): the hitting angles can hide in a window
                   too narrow for any march to land in by luck.

        Every probe is one closed-form flight; probes are bounded by MAX_MARCH/ANGLE_STEPS;
        and an angle is returned only when the forward model agrees to DIST_ACCURACY.
        """
        self._check()                                 # a broken bow is not a missed shot
        if not self.min_dist <= target_distance <= self.max_dist:
            raise self._beyond("get_angle", target_distance)
        target_alt: float = -(self.archer_alt + self.ar_launch_alt)   # arrow to target, up +
        if self.segments == 1 and target_alt <= 0.:
            raise Unreachable("the rising crossing wants the arrow below the target "
                              f"(target_alt {target_alt} m)")
        eye: float = math.atan2(target_alt, target_distance)           # the sight line
        try:
            # Fast path: at short range drag barely moves the no-drag answer, so try it first.
            guess: float = self._vacuum_angle(target_distance, target_alt)
            if abs(guess) < PI_2 - 1e-4:
                hit: float = self._range_of(guess)
                if abs(hit - target_distance) <= self.DIST_ACCURACY:
                    self.get_horizontal_distance(guess)  # fill final_* for the accepted shot
                    return guess
            # Gravity puts the arrow below the line it was launched along, so when the arrow starts
            # above the target line the eye line is the flat end of every branch and a safe place to
            # start scanning.  It is not safe when the arrow starts *below* the target line (target
            # above the archer, segments=2): aiming straight at that target already overshoots, and
            # the flattest hit then lies *flatter* than the eye line, where the distance crosses the
            # target on its way up.  Find that crossing instead of missing it.
            low: float = min(max(eye, -(PI_2 - 1e-4)), PI_2 - 1e-4)
            if self._past(True, target_distance, low):
                low = self._flat_edge(target_distance, low)
            scan: float = low                            # the phases push this up as they fail
            best: float = -1.
            best_angle: float = low
            for reaches in (True, False):          # phase 1 (>= target), then phase 2 (<=)
                span = self._bracket_up(reaches, target_distance, scan,
                                        None if not reaches else guess)
                if span is None:
                    break
                a, b = span
                angle: float = a if a == b else self._refine_angle(reaches, target_distance, a, b)
                probe: float = self._range_of(angle)
                if probe > best:
                    best, best_angle = probe, angle
                if abs(probe - target_distance) <= self.DIST_ACCURACY:
                    self.get_horizontal_distance(angle)  # fill final_* for the accepted shot
                    return angle
                scan = max(scan, a, angle)                # keep scanning above a failed branch
            jump: float | None = self._jump_angle(target_distance, low)
            if jump is not None:
                probe = self._range_of(jump)
                if probe > best:
                    best, best_angle = probe, jump
                if abs(probe - target_distance) <= self.DIST_ACCURACY:
                    self.get_horizontal_distance(jump)    # fill final_* for the accepted shot
                    return jump
            raise _Miss(self.segments, best, best_angle)
        except BadInput as err:
            # The search stepped outside the angles this model will fly: nearer vertical than the
            # guard allows, or a headwind that beats the arrow at that angle.  _check() has already
            # passed, so this is not the caller's setup - it is this crossing running out of flyable
            # angles, and the crossing is refused with the guard's own reason.  Reporting it as
            # BadInput made an unhittable shot look like a broken bow.
            raise Unreachable(f"the {_CROSSING[self.segments]} crossing needs an angle the model "
                              f"will not fly ({err})") from None

    def _flat_edge(self, target: float, hi: float) -> float:
        """An angle just above where distance(angle) grows through `target`, below `hi`.

        distance(angle) grows with the aim up to its maximum, so when aiming along `hi` already
        sends the arrow at least `target` metres, the flattest hitting angle is flatter than
        `hi`.  Bisect "reaches at least target" between nearly straight down and `hi` and hand
        back the *short* side: the phases march up from there and bracket the crossing like any
        other.  Bounded: 2 * BISECT_STEPS probes.
        """
        lo: float = -(PI_2 - 1e-4)
        if self._past(True, target, lo):
            return lo                            # even straight down reaches: nothing to bracket
        a, b = lo, hi                            # invariant: short at a, reaches at b
        for _ in range(2 * BISECT_STEPS):        # bounded bisection
            m: float = 0.5 * (a + b)
            if self._past(True, target, m):
                b = m
            else:
                a = m
            if b - a < ANGLE_EPS:
                break
        return a

    def _jump_angle(self, target: float, low: float) -> float | None:
        """An angle hidden immediately above the sight-line jump, or None.

        distance(angle) is discontinuous: below the *grazing* angle the arrow never comes back
        down to the sight line (distance = 'never'), and immediately above it the distance is
        at its largest.  So the angles that land at a given target can occupy a window a few
        thousandths of a radian wide, right above the jump - and a march of any step size can
        step straight over it.  (Real case: segments=1, archer_alt=-100, target 300 m, where
        the whole window is 0.5122..0.5126 rad.)

        Fix: bisect for the jump, step a hair above it, and root-find on the smooth branch
        that starts there.  Only ever called after the two ordinary phases have failed, so the
        ~40 extra flights are paid for only by the shots that would otherwise be missed.
        """
        top: float = PI_2 - 1e-4
        if self._range_of(low) >= 0. or self._range_of(top) < 0.:   # no jump above `low`
            return None

        def reaches(x: float) -> float:
            return 1. if self._range_of(x) >= 0. else -1.

        a, b = low, top                                  # reaches(a) < 0 <= reaches(b)
        for _ in range(2 * BISECT_STEPS):                # bounded: the grazing angle
            m: float = 0.5 * (a + b)
            if reaches(m) > 0.:
                b = m
            else:
                a = m
            if b - a < 1e-12:
                break
        just_above: float = b
        if not self._past(True, target, just_above):
            return None                                  # even grazing, it falls short
        far: float = just_above
        for _ in range(MAX_MARCH):                       # find where it drops below the target
            far = min(far + max(ANGLE_FLOOR, ANGLE_MARCH * far), top)
            if not self._past(True, target, far):
                return self._refine_angle(True, target, far, just_above)
            if far >= top:
                return None
        return None

    def _past(self, reaches: bool, target: float, angle: float) -> bool:
        """The monotone predicate of one branch: 'this angle already puts the arrow at or
        beyond the target' (reaches) / 'at or short of it' (not reaches).  A flight that
        never meets the sight line answers 'no' to both, which is right on both branches:
        too flat to clear the line, or too flat to have come down yet."""
        dist: float = self._range_of(angle)
        if dist < 0.:
            return False
        return dist >= target if reaches else dist <= target

    def _bracket_up(self, reaches: bool, target: float, low: float,
                    hint: float | None = None) -> tuple[float, float] | None:
        """First interval [a,b] above `low` on which the predicate flips to True, or None.

        Probes grow geometrically (floored at ANGLE_FLOOR), because the answer is 0.007 rad
        for a 10 m target and 1.5 rad for a vertical one: a fixed step is either hopeless or
        wasteful.  `hint` (the no-drag angle) is tried first: when it already overshoots, the
        bracket is [low, hint] and the whole march is skipped.

        The growth factor is small (ANGLE_MARCH) on purpose.  distance(angle) is discontinuous
        - below some angle the arrow stops clearing the sight line and the distance goes from
        hundreds of meters to 'never' in one step - so a legal hitting angle can occupy a
        window only ~0.05 rad wide.  A 0.35 rad probe march stepped straight over one of
        those and answered Unreachable for (segments=1, archer_alt=-100, 300 m), where the
        hitting angle is 0.51334 rad.
        """
        a: float = low
        if self._past(reaches, target, a):
            return a, a
        if hint is not None and a < hint < PI_2 - 1e-4 and self._past(reaches, target, hint):
            return a, hint
        if hint is not None:
            a = max(a, hint)
        for _ in range(MAX_MARCH):                   # bounded march upward
            b: float = min(a + max(ANGLE_FLOOR, ANGLE_MARCH * abs(a)), PI_2 - 1e-4)
            if self._past(reaches, target, b):
                return a, b
            if b >= PI_2 - 1e-4:
                return None
            a = b
        return None

    def _reachable_anchor(self, short: float, reached: float) -> float:
        """Distance at the first angle past `short` that reaches the sight line at all.

        `short` is an angle whose arrow never gets back to the sight line (distance -1) and
        `reached` one that does.  -1 is a search marker, not a distance, so it is useless as a
        secant anchor: a secant pinned to a fabricated 0 crawls (48 passes and still 11 cm off,
        on the shot whose target sits just past the grazing jump).  The grazing angle's own
        distance is a real short shot, so it anchors the secant and the root falls out in 3-4
        passes.  Bounded: 2 * BISECT_STEPS probes.
        """
        a, b = short, reached                      # invariant: a never reaches, b does
        x_b: float = self._range_of(b)
        for _ in range(2 * BISECT_STEPS):          # bounded bisection of the reachable boundary
            m: float = 0.5 * (a + b)
            x_m: float = self._range_of(m)
            if x_m >= 0.:
                b, x_b = m, x_m
            else:
                a = m
            if b - a < 1e-12:
                break
        return x_b

    def _refine_angle(self, reaches: bool, target: float, a: float, b: float) -> float:
        """The end of [a,b] where the predicate turns True: bisection plus a guarded secant.

        The secant interpolates distance(angle), smooth inside a branch, so it usually lands
        in 3-4 probes; bisection is the fallback that cannot be fooled by the jump.  `a` and
        `b` may come in either order - _jump_angle hands them over the jump, where the predicate
        turns True going the other way - so every test here is on the width, not the sign.
        """
        x_a: float = self._range_of(a)
        x_b: float = self._range_of(b)
        if x_a < 0.:
            x_a = self._reachable_anchor(a, b)     # a real short shot, not the -1 marker
        if x_b < 0.:
            x_b = self._reachable_anchor(b, a)
        for _ in range(ANGLE_STEPS):                 # bounded: no while True anywhere
            if abs(b - a) < ANGLE_EPS:
                break
            cand: float = 0.5 * (a + b)              # regula falsi, guarded by bisection
            if x_a >= 0. and x_b >= 0. and x_a != x_b:
                guess: float = a + (target - x_a) * (b - a) / (x_b - x_a)
                if min(a, b) < guess < max(a, b):
                    cand = guess
            x_c: float = self._range_of(cand)        # one flight per pass, used twice below
            if abs(x_c - target) <= 0.5 * self.DIST_ACCURACY:
                return cand                          # good enough for the caller's check
            past: bool = (x_c >= target) if reaches else (0. <= x_c <= target)
            if past:
                b, x_b = cand, x_c
            else:
                a, x_a = cand, x_c                   # keep the -1: the secant gate below reads it
        return b

    def _vacuum_angle(self, d: float, dy: float) -> float:
        """Closed-form no-drag elevation, low branch: a starting probe, always a bit low."""
        v2: float = self.ar_launch_velo ** 2
        disc: float = v2 * v2 - self.gravity_accel * (self.gravity_accel * d * d + 2. * dy * v2)
        if disc < 0. or d <= 0.:
            return 0.
        return math.atan((v2 - math.sqrt(disc)) / (self.gravity_accel * d))

    def _safe_range(self, angle: float) -> float:
        try:
            return self._range_of(angle)
        except (Unreachable, BadInput):
            return -1.

    # ------------------------------------------------------ model diagnostics ----
    def _peak_angle(self) -> float:
        """The launch angle that carries farthest, by bounded golden section over (0, pi/2).

        Searched over the whole (0, pi/2): with the target high above the archer the flattest
        shot that reaches it is a plunging one, well past 45 deg, and capping the search at
        45 deg made max_range answer -1.0 for (segments=1, archer_alt=-100) instead of 319 m.
        """
        lo, hi = 0., PI_2 - 1e-4
        gr: float = (math.sqrt(5.) - 1.) / 2.
        a: float = hi - gr * (hi - lo)
        b: float = lo + gr * (hi - lo)
        fa: float = self._safe_range(a)
        fb: float = self._safe_range(b)
        for _ in range(40):                          # bounded golden section
            if fa < fb:
                lo, a, fa = a, b, fb
                b = lo + gr * (hi - lo)
                fb = self._safe_range(b)
            else:
                hi, b, fb = b, a, fa
                a = hi - gr * (hi - lo)
                fa = self._safe_range(a)
        return a if fa >= fb else b

    def max_range(self) -> float:
        """The model's own maximum range from the archer's altitude, in meters."""
        return self._safe_range(self._peak_angle())

    @property
    def max_dist(self) -> float:
        """The farthest distance the interface takes a question about, in meters.

        Derived from the model - it used to be `612.`, a constant written down for the default
        MAG340 arrow, which quietly became a lie for every other one: a slicker arrow (drag_coef
        1.4e-3, or a calibrate() that lands there) carries 1091 m in this model and the constant
        refused 700 m it could hit with room to spare.  The limit is `max_range()` itself now, so
        nothing hittable is refused for the wrong reason and nothing beyond the possible is
        missed: above it no angle exists by definition.  Between it and the last distance the
        angle search can still verify inside DIST_ACCURACY (about 0.7% below it at the defaults,
        where the range-angle curve is flattest) the search refuses and says which crossing failed
        - the limit is the range of the question, not a promise that an answer exists.

        Deriving it costs 40 closed-form flights (~16 ms) and the object reads it twice per
        question, so it is remembered against FAR_STAMP: the twelve settings that move a
        trajectory.  The read-back is deliberately not in that stamp - answering a question writes
        hit_speed, crossing, note and the rest, and a cache invalidated by its own output is a
        cache that never hits.  When the model cannot bring an arrow back to the sight line at any
        angle (a target high above the archer) there is no far limit to state and it is `inf`, not
        the search's `-1.` marker.  Assign a number to impose your own ceiling, or None to derive.
        """
        if self._far is not None:
            return self._far
        key: tuple[float, ...] = tuple(getattr(self, name) for name in FAR_STAMP)
        if self._far_cache is None or self._far_key != key:
            self._far_cache, self._far_key = self._far_range(), key
        return self._far_cache

    @max_dist.setter
    def max_dist(self, limit: float | None) -> None:
        self._far, self._far_cache, self._far_key = limit, None, None

    def _far_range(self) -> float:
        """`max_range()` with the crossing put back on auto, because the limit is not about shots.

        `max_range()` answers the question `segments` is currently asking: with `segments = 1` it
        returns the rising crossing's own maximum - 7.4 m for the default arrow at level ground -
        and an archer who has asked for the rising shot is still entitled to *ask* about 300 m,
        because auto falls back to the falling crossing and says so in the note. That fallback is
        the whole design of `segments`, so the range of the question is the widest the model can
        answer, whichever crossing ends up answering it.
        """
        keep: int = self.segments
        self.segments = 0                             # type: ignore[assignment]
        try:
            far: float = self.max_range()
        finally:
            self.segments = keep                      # type: ignore[assignment]
        # -1. is _safe_range's marker for "this angle never gets back to the sight line", and
        # max_range() hands it back when *no* angle does - a target high above the archer. It is
        # not a distance: a limit of -1 m would refuse every question with a sentence about a
        # negative range while the search itself is left holding the true one ("at archer_alt -500
        # m no angle of this bow brings an arrow to a target 300 m out"). Where the model cannot
        # state a far limit, none is stated, and the entry points explain the setup instead.
        return far if far > 0. else float("inf")

    def calibrate(self, distance: float, measured_speed: float) -> float:
        """Set drag_coef so the arrow passes `measured_speed` m/s at `distance` m.

        Returns the new drag_coef.  This is what the closed form buys the user: one
        chronograph reading at one distance turns into the model's drag, in milliseconds.
        """
        if not self.min_dist <= distance <= self.max_dist or measured_speed <= 0.:
            raise BadInput(f"{distance=}, {measured_speed=}")
        lo, hi = 0., 20.
        for _ in range(40):                          # bounded bisection on the drag
            mid: float = 0.5 * (lo + hi)
            self.drag_coef = mid
            try:
                self.get_angle(distance)             # sets hit_speed at the sight line
                speed: float = self.hit_speed
            except (Unreachable, BadInput):
                speed = 0.                           # cannot even get there: too much drag
            if speed > measured_speed:               # arrow still too fast: add drag
                lo = mid
            else:
                hi = mid
        self.drag_coef = 0.5 * (lo + hi)
        return self.drag_coef

    # ------------------------------------------------------------- interfaces ----
    def interactive(self) -> None:
        """Simple textual user interface: [a=_] [d=_] [s=_] [l=_] [w=_] [h] [q].

        One line is a state, not a script: the settings on it are applied before the question on
        it is answered, whatever order they are typed in, and the `=` is optional (`d50` and
        `d=50` both work) because the two interfaces used to disagree about it and typing the
        wrong one read as an arithmetic error.  What is printed is what you asked, then what the
        arrow is set to, then what you get - question, state, answer, numbers.
        """
        msg: str = (
            f"arrow.py({VERSION}):\n"
            f"Angle[rad]=(-pi/2,+pi/2)  Distance[m]=(+{self.min_dist:g},+{self.max_dist:g})\n"
            f"Segments=0|1|2|3 (0=auto, 3=lofted)  archer_aLt[m]  Wind[m/s, + = tail]  Q=Quit\n"
            f"enter : [a=_] [d=_] [s=_] [l=_] [w=_] [h] [q] <Enter>\n"
            f"enter : ")
        while True:
            try:
                cmdl: list[str] = input(msg).strip().lower().split()
            except EOFError:
                return
            if any(c in QUIT_KEYS for c in cmdl):
                return
            if any(c in HELP_KEYS for c in cmdl):
                print(f"\n{help_text(self)}\n")
                continue
            try:
                cmdd: dict[str, float] = parse_keys(cmdl)
                apply_settings(self, cmdd)
                rows: list[Row] = []
                if "a" in cmdd and "d" in cmdd:
                    rows.append(("input :", "launch angle", "rad", f"{cmdd['a']}"))
                    rows.append(("input :", "flight distance", "m", f"{cmdd['d']}"))
                    settings_rows(self, rows)
                    gap: float = self.get_vertical_distance(cmdd["a"], cmdd["d"])
                    rows.append(("output:", "vertical error", "m", f"{gap:.3f}"))
                elif "a" in cmdd:
                    rows.append(("input :", "launch angle", "rad", f"{cmdd['a']}"))
                    settings_rows(self, rows)
                    rows.append(("output:", "flight distance", "m",
                                 f"{self.get_horizontal_distance(cmdd['a']):.3f}"))
                elif "d" in cmdd:
                    rows.append(("input :", "required distance", "m", f"{cmdd['d']}"))
                    settings_rows(self, rows)
                    rows.append(("output:", "launch angle", "rad",
                                 f"{self.get_angle(cmdd['d']):.4f}"))
                else:
                    settings_rows(self, rows)       # settings only, or an empty line: re-state
                asked: bool = "a" in cmdd or "d" in cmdd
                if asked:
                    shot_rows(self, rows)
                print(answer_block(rows))
                if asked and self.note:
                    print(f"{INDENT}note: {self.note}")
                print()
            except (ValueError, OverflowError) as err:
                name: str = str(err.__class__).replace("__main__.", "").replace("'", "")
                print(f"{INDENT}{name.replace('class', 'error')} {err!s}\n")


def main() -> None:
    """No args: the REPL. With args: one shot, e.g.  arrow.py d=50 l=10 s=2 w=3

    One grammar for both interfaces: a= d= s= l= w=, the = optional, any order - the settings are
    applied before the question is answered.  -h / --help prints the help and exits.
    """
    import sys
    arr: Arrow = Arrow()
    args: list[str] = sys.argv[1:]
    if any(arg in HELP_KEYS for arg in args):
        print(help_text(arr))
        return
    if not args:
        arr.interactive()
        return
    try:
        nums: dict[str, float] = parse_keys(args)
        apply_settings(arr, nums)                   # settings first, the question afterwards
    except BadInput as err:
        raise SystemExit(str(err)) from None
    if "a" in nums:
        print(f"angle {nums['a']} rad -> distance "
              f"{arr.get_horizontal_distance(nums['a']):.3f} m")
    if "d" in nums:
        print(f"distance {nums['d']} m -> angle {arr.get_angle(nums['d']):.5f} rad")
    if "a" in nums or "d" in nums:
        rows: list[Row] = []
        shot_rows(arr, rows)
        print(answer_block(rows))
        if arr.note:
            print(f"{INDENT}note: {arr.note}")


if __name__ == "__main__":
    main()
