# Arrow

Closed-form external ballistics for an arrow shot from a crossbow. Give it a distance and it
returns the launch angle; give it an angle and it returns the distance; give it both and it
returns how far high or low the arrow arrives. It also reports impact speed, kinetic energy,
time of flight, impact angle and trajectory height.

The whole solver is **one file, standard library only, and it never steps through time**. Every
number below was produced by running this file, not copied from a textbook. What was checked, and
how, is in [Accuracy](#accuracy--two-checks).

```
$ python3 arrow.py d=50
distance 50.0 m -> angle 0.02416 rad
        crossing [-]   = 2 (falling crossing)
        hit_speed [m/s]= 98.359
        hit_energy [J] = 109.708
        hit_time [s]   = 0.495
        hit_angle [rad]= -0.0240
        hit_x [m]      = 50.000
        hit_y [m]      = -0.000
        apex [m]       = 0.289
```

Nine numbers describe the shot the model just made, and they are printed as a table: labels in one
column, the `=` in one column. `crossing` says which shot they describe — here the arrow met the
sight line coming down — and `hit_x`/`hit_y` are where it met it, `hit_speed`/`hit_energy` how it
arrived, `hit_time` how long it took, `hit_angle` at what angle, `apex` the top of the arc.

---

## Contents

- [What makes it different](#what-makes-it-different)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [The model](#the-model)
- [Which crossing the arrow takes](#which-crossing-the-arrow-takes)
- [Accuracy — two checks](#accuracy--two-checks)
- [Library API](#library-api)
- [Worked numbers](#worked-numbers)
- [What the model does not do](#what-the-model-does-not-do)
- [The numerical contract](#the-numerical-contract)
- [Checks you can run yourself](#checks-you-can-run-yourself)
- [Version](#version)
- [Files, testing, license](#files-testing-license)

---

## What makes it different

**No time stepping.** A conventional external-ballistics program integrates the equations of
motion with a time step and iterates until the answer stops moving. This one does not. The
quadratic-drag problem has a classical first integral (the hodograph form the module header
names), so the speed at every point of the flight comes out of a closed-form expression, and
position and time are single definite integrals over the flight path. They are evaluated by a
fixed 8-node Gauss-Legendre rule on panels of bounded width — a constant number of function
evaluations, no error accumulation, no stability limit, no step size to choose and no tolerance
to tune.

**Wind is exact, not fitted.** A steady wind is a Galilean shift: the trajectory is solved in
the air frame and displaced by `wind * t`. Verified against an independent integration that
computes drag from the relative velocity — they agree to 1e-13 m.

**Fast enough to be a library, small enough to read.** One file, `math` and `typing`, no
dependencies, nothing to install. A full inverse solve (find the launch angle for a distance)
costs about 1.4 ms on this host.

**It says when it cannot answer, and says what it tried.** A target that cannot be hit raises
`Unreachable`, and the message names both crossings and how close each got:

```
in get_angle: 3.0 m cannot be hit at archer_alt 0.0 m: the falling crossing got no closer to it
than 7.4 m (at 0.0068 rad)
```

A parameter the model cannot represent raises `BadInput`, a subclass of `Unreachable`, and it
survives the crossing search instead of being reported as a missed shot.

**You ask for a target, not for a branch of the trajectory.** The arrow crosses the sight line
twice — once climbing, once falling — and you do not have to know which one you want: `segments`
defaults to auto, the solver asks the rising crossing first and the falling one second, and
`crossing` says which answered. See
[Which crossing the arrow takes](#which-crossing-the-arrow-takes).

---

## Requirements

Python **3.10 or newer**, standard library only. (The minimum comes from PEP 604 `X | None`
annotations, which are evaluated at import time, and `zip(..., strict=True)`.) Developed and
tested on CPython 3.14.4. Nothing to install, no build step, no network access at run time.

---

## Quick start

### One shot from the command line

```
python3 arrow.py [a=<rad> | d=<m>] [s=<0|1|2>] [l=<m>] [w=<m/s>]
python3 arrow.py -h
```

`-h` (or `--help`) prints the whole interface — the keys, the four shots it can give you, what is
read back, five examples and the model's limits — and exits.

```
$ python3 arrow.py d=50                 # required distance -> launch angle
distance 50.0 m -> angle 0.02416 rad
crossing [-]   = 2 (falling crossing)

$ python3 arrow.py d=3                  # too close for a falling shot: it answers anyway
distance 3.0 m -> angle 0.00970 rad
crossing [-]   = 1 (rising crossing)

$ python3 arrow.py a=0.05               # launch angle -> distance
angle 0.05 rad -> distance 101.224 m
crossing [-]   = 2 (falling crossing)

$ python3 arrow.py d=612                # near the far end of what the default arrow carries
distance 612.0 m -> angle 0.63208 rad
hit_time [s]   = 10.639

$ python3 arrow.py l=10 s=2 w=3 d=50    # archer 10 m above the target, 3 m/s tailwind
distance 50.0 m -> angle -0.17356 rad
crossing [-]   = 2 (falling crossing)

$ python3 arrow.py s=2 d=120 l=-50      # the plunging shot: an arrow over a wall
distance 120.0 m -> angle 1.46834 rad
hit_time [s]   = 16.457
apex [m]       = 311.301
note: this is the high shot, apex 311.3 m; the rising crossing hits the same target flatter and faster, at +0.45697 rad in 1.383 s - take that one unless something is in the way

$ python3 arrow.py s=1 d=300            # the flat shot cannot get that far, and says so
distance 300.0 m -> angle 0.17461 rad
crossing [-]   = 2 (falling crossing)
note: you asked for the rising crossing and it cannot hit this target (the rising crossing cannot get that far); this is the falling crossing's answer
```

`a=` and `d=` are the question; `s=` selects the crossing — `0` (the default) is auto, `1` the
rising crossing, `2` the falling one; `l=` is the archer's altitude relative to the target, `w=`
the wind, `+` being a tailwind. The settings are applied before the question is answered,
whatever order you type them in, so `d=50 l=10 s=2 w=3` and `l=10 s=2 w=3 d=50` give the same
shot. Asking for a crossing that cannot reach is not an error: you get the other crossing's
answer and a `note:` line saying what you asked for and why it is not what you got — see
[Which crossing the arrow takes](#which-crossing-the-arrow-takes).

### The REPL

```
$ ./arrow.py
arrow.py(3.1.0):
Angle[rad]=(-pi/2,+pi/2)  Distance[m]=(+0.01,+617.732)
Segments=0|1|2|3 (0=auto, 3=lofted)  archer_aLt[m]  Wind[m/s, + = tail]  Q=Quit
enter : [a=_] [d=_] [s=_] [l=_] [w=_] [h] [q] <Enter>
enter : d=50 l=-50
input : required distance [m]= 50.0
        segments [-]         = 0
        archer alt [m]       = -50.00
        wind [m/s]           = 0.00
output: launch angle [rad]   = 0.8103
        crossing [-]         = 1 (rising crossing)
        hit_speed [m/s]      = 91.394
        hit_energy [J]       = 94.720
        hit_time [s]         = 0.727
        hit_angle [rad]      = 0.7584
        hit_x [m]            = 49.999
        hit_y [m]            = 0.000
        apex [m]             = 0.000

arrow.py(3.1.0):
Angle[rad]=(-pi/2,+pi/2)  Distance[m]=(+0.01,+617.732)
Segments=0|1|2|3 (0=auto, 3=lofted)  archer_aLt[m]  Wind[m/s, + = tail]  Q=Quit
enter : [a=_] [d=_] [s=_] [l=_] [w=_] [h] [q] <Enter>
enter : q
```

The answer is read in the order it is asked in: what you typed, then what the arrow is set to,
then the answer, then the numbers of that shot — one column for the labels, one for the `=`.
`crossing` comes first among the numbers because it says *which shot* the rest of them describe.

One line is a state, not a script: every key on it is applied, then the question on it is
answered, whatever order they are written in. `h` prints the help, `q` quits, end-of-input quits.
The `=` is optional — `d50`, `a0.05`, `s2`, `l10`, `w-5` still work exactly as `d=50` does — and
settings stick until you change them, so `l-25 w4 d180` is one line and `d180` after it is the
same shot. `s0` puts the crossing back on auto, `s3` asks for the lofted route — [the other shot
at the same target](#shooting-over-a-wall).

### As a library

```python
import math
from arrow import Arrow, Unreachable

arr = Arrow()                 # MAG340 defaults, level shot, falling crossing
angle = arr.get_angle(91.44)  # 100 yards -> radians
print(f"{angle:.5f} rad = {math.degrees(angle):.3f} deg")
print(f"impact {arr.hit_speed:.1f} m/s, {arr.hit_energy:.1f} J, "
      f"{arr.hit_time:.2f} s, apex {arr.apex:.2f} m")

arr.wind = 5.0                # tailwind, m/s
arr.archer_alt = 12.0         # archer 12 m above the target
print(arr.get_angle(200.0))

try:
    arr.get_angle(700.0)      # past what this arrow can carry at these settings
except Unreachable as err:
    print(err)
```

```
0.04489 rad = 2.572 deg
impact 94.2 m/s, 100.6 J, 0.93 s, apex 1.04 m
0.044151892867459026
in get_angle: 700 m is past the limit 657.3 m (this arrow carries at most 657.3 m at these settings; the nearest question the model takes is 0.01 m)
```

---

## The model

A point mass in a constant gravitational field with quadratic drag always opposite the velocity
— the standard external-ballistics model, solved exactly instead of marched.

```
d vx/dt = -k |v| vx
d vy/dt = -g - k |v| vy            k = 0.5 rho Cd A / m      [1/m]
```

The whole solution rests on one observation: drag is parallel to the velocity, so if `theta` is
the flight-path angle then the speed satisfies a **linear** equation in `u_ = 1/v**2`,

```
du_/dtheta + 2 tan(theta) u_ = -(2k/g) sec(theta)
```

whose solution is the first integral of the quadratic-drag problem,

```
1/v**2 = cos**2(theta) * [ 1/(v0**2 cos**2(alpha)) + (k/g) (G(sin alpha) - G(sin theta)) ]
G(s)   = s/(1-s**2) + atanh(s)
```

Two limits check it for free: `k -> 0` gives `v = v0 cos(alpha)/cos(theta)` (horizontal speed
conserved) and `theta -> -pi/2` gives `v = sqrt(g/k)`, terminal speed.

Position and time are then quadratures along the path. The natural variable `theta` is dreadful
near a vertical plunge — a shot that ends falling straight down spends its last hundred metres
in a sliver of `theta` of width 1e-5 — so the integrals are written in

```
u = asinh(tan(theta))    <=>   tan(theta) = sinh(u), cos(theta) = 1/cosh(u), sin(theta) = tanh(u)
```

in which the pole at `theta = -pi/2` runs off to `u = -infinity` and everything is smooth, with
no trigonometry left beyond `sinh`/`cosh` of moderate arguments:

```
P(u)  = 1 / ( c - (k/g)(sinh(u) cosh(u) + u) ),      v**2 = P(u) cosh**2(u)
x     = (1/g) Int P cosh(u) du
y     = y0 + (1/g) Int P sinh(u) cosh(u) du
t     = (1/g) Int sqrt(P) cosh(u) du
```

with the analytic derivatives `dx/du = -P cosh(u)/g`, `dy/du = -P sinh(u) cosh(u)/g` that root
finding uses. Setting `k = 0` collapses these to the textbook vacuum formulas term by term
(the agreement is checked below), and the drag coefficient, mass, diameter and vane area of a real
arrow enter through `k` alone.

**Wind.** A steady wind is a Galilean shift. The flight is solved in the air frame from the
launch velocity `(v0 cos(a) - wind, v0 sin(a))` and the ground position is `x_air + wind * t`.
Nothing is fitted and nothing is linearised.

**Which crossing.** The arrow meets the sight line climbing and, later, falling. `segments = 2`
is the ordinary shot (the arrow comes *down* onto the target), `segments = 1` is the rising
crossing, for targets above the archer. The default is `0` = auto, which asks them in that order;
see [Which crossing the arrow takes](#which-crossing-the-arrow-takes).

The module docstring at the top of `arrow.py` states the same derivation in full, with the
numbers behind each claim.

---

## Which crossing the arrow takes

The arrow is released a little below the archer's eye (`ar_launch_alt`, 2.5 cm) and the sight
line is the horizontal plane through that eye, so a shot meets the sight line **twice**: once
climbing, once falling. Which one is the shot you want has a name — `segments` — and its default
is `0`, meaning *ask both*.

`segments` says which crossing you want and `crossing` says which one answered. Nothing on offer
is a dead end: a crossing that cannot reach hands the shot to the other one and writes an
explanation in `note`, because the shot you asked for and the shot you get are two different
arrows and saying which is the whole point.

| `segments` | when that crossing can hit | when it cannot |
|---|---|---|
| `0` auto | the rising one if it can reach, else the falling one — the **faster route** | (nothing to fall back from) |
| `1` rising | the rising shot; `note` names the falling one too | the falling shot, note quoting the rising crossing's own refusal |
| `2` falling | the falling shot — the wall-clearing one; note names the rising one | the rising shot, note quoting the falling crossing's own refusal |
| `3` lofted | the **steepest** angle that hits — the high shot over what is in the way | refuses below about 0.12 m, where no lofted shot exists |

`Unreachable` comes only when **neither** crossing can hit, and its message says what each one did.

`3` is not a crossing. It asks the same model for the steepest angle that hits, which for a target
level with the arrow sits on the *same* crossing as the flat one — which is exactly why `1` and `2`
cannot reach it. See [Shooting over a wall](#shooting-over-a-wall).

Auto picks the rising crossing first because the arithmetic says that is the quick arrow, not the
steep one. Of the 120 geometries tried — archer altitudes from 200 m below the target to 300 m
above it, targets 1 m to 612 m out — 29 have both crossings, and in every one of those 29 the
rising shot is the flatter **and** the faster:

| target | rising (`crossing=1`) | falling (`crossing=2`) | faster |
|---|---|---|---|
| 120 m out, 50 m above the archer | 0.456972 rad, **1.383 s** | 1.468337 rad, 16.457 s | rising, 12x |
| 300 m out, 100 m above | 0.513337 rad, **3.959 s** | 1.286643 rad, 15.047 s | rising, 4x |
| 300 m out, 200 m above | 0.813475 rad, **5.196 s** | 1.250284 rad, 12.976 s | rising, 2.5x |

For level targets there is usually only one crossing to have, and the two hand over exactly where
the geometry says they must. The two crossings of the sight line obey
`x1 * x2 = v0^2 * 2 * target_alt / g`, so the rising crossing can never exceed
`v0 * sqrt(2 * target_alt / g)` — **7.399 m** for the default arrow at level ground:

| target, level | rising crossing | falling crossing | auto returns |
|---|---|---|---|
| 1 m | 0.025452 rad | refuses | the rising shot |
| 3 m | 0.009703 rad | refuses | the rising shot |
| 6 m | 0.006923 rad | refuses | the rising shot |
| 7.4 m | refuses | 0.006776 rad | the falling shot |
| 50 m | refuses | 0.024161 rad | the falling shot |
| 612 m | refuses | 0.632078 rad | the falling shot |

That bound is also what keeps auto cheap: an ordinary flat shot is screened out of the rising
search and costs one solve, 1.50 ms, instead of two. (With any wind the bound is not trusted and
both crossings are searched, because a tailwind shifts the ground frame. Asking for a forced
crossing costs both solves, about 35 ms for the high shot at 120 m — the price of being told what
the alternative was.)

Every distance worth asking is hittable: **1224 distances from `min_dist` to 612 m, stepped by
0.5 m, were asked of the auto solver and every one of them was hit**, the worst landing
error 0.027 m against the model's own acceptance of 0.05 m, slowest call 10.5 ms. The same sweep was
then run over `segments = 0`, `1` and `2` — 243 shots across nine archer altitudes — and every
angle handed back was checked against the independent RK4 march on the crossing that answered:
**none landed further from its target than 0.0391 m**, inside `DIST_ACCURACY` (12 of the 243 are
legitimate refusals, each naming both crossings).

### Shooting over a wall

`segments = 2` is not decoration. Archer on a slope, target 120 m out and 50 m below the eye, and
a wall standing halfway between:

```python
from arrow import Arrow
# The archer is on a slope: the target is 120 m away and 50 m below the eye, and a wall stands
# halfway. segments=2 is the only setting that gets an arrow over it.
a = Arrow()
a.archer_alt, a.segments = -50.0, 2
high = a.get_angle(120.0)
print(f"s=2: {high:.5f} rad ({high * 57.2958:.1f} deg), apex {a.apex:.1f} m, "
      f"{a.hit_time:.1f} s of flight")
print(f"     over the wall at 60 m: {a.get_vertical_distance(high, 60.0):+.2f} m")
b = Arrow()
b.archer_alt, b.segments = -50.0, 1
low = b.get_angle(120.0)
print(f"s=1: {low:.5f} rad ({low * 57.2958:.1f} deg), {b.hit_time:.2f} s of flight")
print(f"     over the wall at 60 m: {b.get_vertical_distance(low, 60.0):+.2f} m - under it")
```

```
s=2: 1.46834 rad (84.1 deg), apex 311.3 m, 16.5 s of flight
     over the wall at 60 m: +307.81 m
s=1: 0.45697 rad (26.2 deg), 1.38 s of flight
     over the wall at 60 m: -22.66 m - under it
```

Same target, same bow, 330 m of difference at the wall — and `get_vertical_distance(angle, x_wall)`
is the whole of what clearing a wall asks for: the height of the arrow where the wall is. It
follows `crossing`, so a forward call after a `get_angle` describes the shot you were just given.

**The other shot at the same target: `segments = 3`.** `get_angle` returns the *flattest* hit, and
for a target level with the archer both of a target's two solutions are falling crossings — the
steep twin is on the same crossing as the flat one, so `segments = 2` does not reach it either.
`steepest_angle(distance)` does, and `segments = 3` asks for it through `get_angle`. For a target
300 m out it is **1.307970 rad (74.9 deg)**, clearing 339 m at half range where the flat shot
scrapes along at 0.174606 rad; for one 60 m out, **1.521556 rad (87.2 deg)**.

```python
from arrow import Arrow
# The same bow, a level target, and something growing at half distance. The flat shot is in it;
# the lofted one goes over. steepest_angle is the only way to be asked for it.
arw = Arrow()
flat = arw.get_angle(60.0)
print(f"flat:   {flat:.5f} rad ({flat * 57.2958:.1f} deg), {arw.hit_time:.2f} s of flight, "
      f"{arw.hit_speed:.1f} m/s at the target")
loft = arw.steepest_angle(60.0)
print(f"lofted: {loft:.5f} rad ({loft * 57.2958:.1f} deg), {arw.hit_time:.2f} s of flight, "
      f"apex {arw.apex:.1f} m")
print(f"        height at a hedge 30 m out: lofted {arw.get_vertical_distance(loft, 30.0):+.1f} m,"
      f" flat {arw.get_vertical_distance(flat, 30.0):+.2f} m")
print(f"note:   {arw.note}")
```

```
flat:   0.02902 rad (1.7 deg), 0.60 s of flight, 97.3 m/s at the target
lofted: 1.52155 rad (87.2 deg), 17.25 s of flight, apex 364.1 m
        height at a hedge 30 m out: lofted +362.6 m, flat +0.43 m
note:   this is the lofted shot: 87.2 deg, apex 364.1 m; the flat route hits the same target at +0.02902 rad in 0.598 s - take that one unless something is in the way
```

It is the same closed form on the same branch, not new physics: on the far side of the angle of
maximum range the distance falls monotonically down to the near-vertical landing point, so one
bracketed bisection walks down to the root, and the angle comes back only when the model's own
forward call agrees it lands within `DIST_ACCURACY`. The read-back block describes the lofted shot
and `crossing` still says which crossing answered — falling, for a level target. `note` is
written with the flat route's angle and time beside it, because the two differ by 17 seconds of
flight and that is the archer's decision, not the solver's.

At the short end there is genuinely nothing to find: below about **0.12 m** even an arrow fired at
the model's vertical limit still lands past the target, and `steepest_angle` refuses with that
number in the message rather than returning an angle it cannot fly.

Up hill (target above the arrow's release, `l` negative) the two solutions sit on different
crossings, and `segments = 2` reaches the lofted one directly, as above.

Two more things worth knowing:

- **A forward question asks about a crossing, not a target.** `get_horizontal_distance(angle)`
  and `get_vertical_distance(angle, x)` take an angle, and with `segments = 0` they follow
  `crossing`, the crossing the last `get_angle` answered on; with no shot asked for yet they use
  the falling one, which is the range of the shot. Set `segments` explicitly to ask about one
  crossing regardless: the same angle can be a hit on one crossing and 80 m short on the other.
- **The deliberate high shot is still there.** Force `segments = 2` and you can have the plunge:
  at `pi/2 - 0.001` the arrow lands 1.223 m out having climbed to a **365 m apex** in 17.3 s of
  flight. Physically a hit, usefully — no. That branch is why `segments` is an override and not
  a deletion.

---

## Accuracy — two checks

**Second solver.** The closed form against an independent fixed-step RK4, written from the ODE
alone — drag from the relative velocity, so no shared trick. Same event: the crossing of the sight
line. They agree to better than 1e-9 of the distance on every shot below, to ~1e-15 on the easy
ones. Arithmetic agreeing with arithmetic.

| shot | quantity | this model | independent RK4 | difference |
|---|---|---|---|---|
| level, 18.7 m (a = 0.01 rad) | distance | 18.72417 m | 18.72417 m | 6e-13 m |
| level, 101.2 m (a = 0.05 rad) | distance | 101.22391 m | 101.22391 m | 7e-14 m |
| level, 483.5 m (a = 0.35 rad) | distance | 483.45766 m | 483.45766 m | 3e-09 m |
| level, 612 m (a = 0.63208 rad) | distance | 612.02233 m | 612.02233 m | 8e-08 m |
| level, 612 m | time of flight | 10.638507 s | 10.638507 s | 1e-09 s |
| level, 612 m | impact speed | 62.09927 m/s | 62.09927 m/s | 3e-09 m/s |
| target 50 m up at 120 m, plunging at 84 deg | distance | 120.00043 m | 120.00043 m | 2e-07 m |
| target 50 m up at 120 m, rising (`segments=1`) | distance | 119.98207 m | 119.98207 m | 2e-12 m |
| 100 m with 12 m/s tailwind | distance | 100.00017 m | 100.00017 m | 6e-14 m |
| 100 m with 12 m/s headwind | distance | 100.00062 m | 100.00062 m | 1e-14 m |

The hardest cases are in there: the 612 m shot, which takes 10.6 s and ends at 49 degrees below
the horizontal, and the 84 degree plunging shot, which lands on the sight line at almost
terminal speed after 16.5 s — exactly where the `theta` formulation loses all resolution and
where a time-stepped solver needs a very small step.

**Vacuum limit.** With `drag_coef = 0` and the arrow launched exactly on the sight line, the
model reproduces the textbook range `v0^2 sin(2a)/g` to `3.4e-13 m` — machine epsilon — at
angles from 0.02 rad to 45 degrees.

**Round trip.** For every distance tried at level ground, `get_horizontal_distance(get_angle(d))`
came back within **0.023 m**, inside the model's own acceptance of `DIST_ACCURACY = 0.05 m`:

| target | angle | forward check | impact speed | energy | time | apex |
|---|---|---|---|---|---|---|
| 10 m | 0.007100 rad (0.41 deg) | 10.002 m | 102.55 m/s | 119.3 J | 0.097 s | 0.00 m |
| 25 m | 0.012621 rad (0.72 deg) | 25.000 m | 100.96 m/s | 115.6 J | 0.244 s | 0.06 m |
| 50 m | 0.024161 rad (1.38 deg) | 50.000 m | 98.36 m/s | 109.7 J | 0.495 s | 0.29 m |
| 91.44 m (100 yd) | 0.044887 rad (2.57 deg) | 91.440 m | 94.20 m/s | 100.6 J | 0.927 s | 1.04 m |
| 100 m | 0.049356 rad (2.83 deg) | 100.000 m | 93.37 m/s | 98.9 J | 1.018 s | 1.26 m |
| 182.88 m (200 yd) | 0.096006 rad (5.50 deg) | 182.880 m | 85.70 m/s | 83.3 J | 1.954 s | 4.67 m |
| 200 m | 0.106473 rad (6.10 deg) | 200.000 m | 84.21 m/s | 80.4 J | 2.159 s | 5.71 m |
| 300 m | 0.174606 rad (10.00 deg) | 300.000 m | 76.20 m/s | 65.8 J | 3.455 s | 14.67 m |
| 400 m | 0.258893 rad (14.83 deg) | 400.001 m | 69.46 m/s | 54.7 J | 4.971 s | 30.43 m |
| 500 m | 0.371725 rad (21.30 deg) | 500.000 m | 64.31 m/s | 46.9 J | 6.858 s | 58.07 m |
| 612 m | 0.632078 rad (36.22 deg) | 612.022 m | 62.10 m/s | 43.7 J | 10.639 s | 140.11 m |

**Arrows.** The model was shot, by the author: many arrows at 5 m to 50 m against the 2.x program
— same equations, marched rather than closed-form. It agreed. One shooter, short range, nothing
logged. At those distances a 10% error in `drag_coef` is worth 0.2 m; at 200 m, 2.6 m; at 400 m,
10 m. So the shooting rules out a gross error and proves little about drag. For the long end,
[`calibrate()`](#calibration).

---

## Library API

### `Arrow()`

Every physical parameter is a plain attribute; set it and call again. The defaults are the
standard configuration of a **MAG340** crossbow with 350 grain arrows.

| attribute | default | unit | meaning |
|---|---|---|---|
| `ar_launch_velo` | 103.632 (= 340 fps) | m/s | arrow speed leaving the string |
| `ar_mass` | 0.0226796185 (= 350 gr) | kg | arrow mass |
| `ar_dia` | 0.0087884 (= 0.346 in) | m | shaft diameter |
| `ar_vane_n` | 3 | – | number of vanes |
| `ar_vane_height` | 0.015 | m | vane height |
| `ar_vane_thick` | 0.0008 | m | vane thickness |
| `drag_coef` | 0.4 | – | drag coefficient (shape + friction + induction) |
| `air_dens` | 1.225 | kg/m^3 | air density |
| `gravity_accel` | 9.81 | m/s^2 | gravity |
| `wind` | 0.0 | m/s | steady wind, `+` = tailwind (air moves toward the target) |
| `archer_alt` | 0.0 | m | archer's altitude relative to the target |
| `ar_launch_alt` | -0.025 | m | arrow height above the sight line at launch (eye-to-arrow) |
| `segments` | 0 | – | which shot you want out of the ones that hit: 0 = auto (the faster route), 1 = rising, 2 = falling, 3 = lofted (`steepest_angle`, the steep twin of the flat shot); a crossing that cannot reach falls back to the other and says so in `note` |
| `min_dist` | 0.01 | m | below this, questions are refused: the model has nothing to say that precisely |
| `max_dist` | derived | m | above this, questions are refused. It is `max_range()` — the farthest this arrow, at these settings, can carry — not a number written down: 617.7 m for the default MAG340, 1091 m for a slicker `drag_coef`, 666 m with an 8 m/s tailwind. Set a number to impose a smaller ceiling, `None` to derive again |

The frontal area used for drag is the shaft disc plus `ar_vane_n * ar_vane_height *
ar_vane_thick`, so `k` (and with it the terminal speed) responds to a vane change as it should.

### Inputs in, outputs out

```
arr.get_angle(target_distance)                     -> launch angle [rad]
arr.steepest_angle(target_distance)                -> the steepest angle that hits [rad]
arr.get_horizontal_distance(launch_angle)          -> distance to the sight line [m]
arr.get_vertical_distance(launch_angle, distance)   -> signed vertical error at that x [m]
```

| output | unit | meaning |
|---|---|---|
| `hit_angle` | rad | the angle it **arrives** at (flight-path angle, ground frame), negative = descending — not the `launch_angle` you passed in |
| `hit_speed` | m/s | speed there (ground frame) |
| `hit_energy` | J | kinetic energy there, `0.5 m v^2` |
| `hit_time` | s | release to there |
| `hit_x` | m | horizontal distance travelled |
| `hit_y` | m | height of the arrow above the sight line at `hit_x` |
| `apex` | m | highest point of the arc — of the flown part of it, so a rising crossing reports the height it ended at |
| `crossing` | – | which crossing answered the last call: 1 rising, 2 falling |
| `note` | – | what the shot you asked for said about its own choice: empty when you got exactly that, otherwise the refusal it worked around, the flatter route you are not taking, or what the lofted shot costs |

The read-back is filled by whichever call returned the shot, so read it after the call, on the
same object. "There" is the point that call was asked about: for `get_angle` and
`get_horizontal_distance` it is where the flight meets the sight line; for
`get_vertical_distance(angle, x)` it is the point at the `x` you named, which is usually mid-flight
— asking how high the arrow is at 120 m on a shot that lands at 574 m leaves `hit_x` at 120.0 and
`hit_time` at the time to get there. `get_angle` and `steepest_angle` fill it for the shot they
accepted, and `note` is theirs alone: a forward call updates `crossing` to the crossing it used but
leaves the note that explained the angle you are holding.

### Diagnostics

```
arr.drag_k()            -> float   # 1.0441950e-03 1/m with the defaults
arr.terminal_speed()    -> float   # 96.927 m/s with the defaults  (sqrt(g/k))
arr.max_range()         -> float   # 617.73 m from level ground
```

`max_range()` searches the whole elevation range, not just below 45 degrees, because with the
target high above the archer the flattest reaching shot is a plunging one well past 45.

### Calibration

```
arr.drag_coef = arr.calibrate(distance=100.0, measured_speed=88.625)
```

Sets `drag_coef` so the arrow passes `measured_speed` at `distance` metres, and returns the new
value. One chronograph reading at one distance becomes the model's drag: from a synthetic
"truth" at `drag_coef = 0.6` (which gives 88.6250 m/s at 100 m) it recovered **0.600000** in
80 ms.

### Errors

```
ValueError
└── Unreachable     # that target cannot be hit: check segments, archer_alt, angle, distance, wind
    └── BadInput    # a parameter is outside what the model can represent (a setup error)
```

`BadInput` is a subclass of `Unreachable` on purpose: code that already catches `Unreachable`
keeps working, and code that wants to tell "that target cannot be hit" from "that is not a
question the model can answer" still can. Messages carry the offending values, e.g.

```
Unreachable: in get_angle: 0.01 m cannot be hit at archer_alt -200.0 m: the rising crossing needs
             an angle the model will not fly (launch angle too close to vertical: angle=
             1.5707514173185602); the falling crossing needs an angle the model will not fly (...)
Unreachable: in get_angle: 650.0 m cannot be hit at archer_alt 0.0 m: the rising crossing cannot
             get that far; the falling crossing never gets back to the sight line at any angle
Unreachable: in get_angle: 700 m is past the limit 617.7 m (this arrow carries at most 617.7 m at
             these settings; the nearest question the model takes is 0.01 m)
BadInput:    segments must be 0 (auto), 1 (rising), 2 (falling) or 3 (lofted), got 4
BadInput:    wind 400.0 m/s is at or beyond the launch speed 103.632 m/s: no angle gets the arrow
             forward
BadInput:    headwind 100.0 m/s beats the arrow at angle 1.50000 rad   # a forward call's own angle
BadInput:    ar_mass, gravity_accel and ar_launch_velo must be positive
```

A distance outside `min_dist..max_dist` names the limit and what the arrow can carry at these
settings, because `target_distance=700.0` told an archer nothing about what to ask instead. A
distance inside it that cannot be hit reports what each crossing managed — "got no closer to it
than 7.4 m", or "never gets back to the sight line at any angle" — rather than the `-1.0 m` the
search uses internally for "not a distance", and it is raised only when *both* crossings have
refused, since one refusing is a fallback and not a failure.

Two kinds of `BadInput` survive to the caller and they are deliberately kept apart. One is the
setup the air cannot support: a wind at or beyond the launch speed leaves no angle to search, so
`_check` says so before any probing and it reads as a broken bow, not a missed shot. The other is
an angle the *caller* handed to a forward call that the model will not fly. An angle the *search*
invented is different: when the branch walk steps past vertical, that crossing has run out of
flyable angles, and it is refused with the guard quoted ("needs an angle the model will not
fly") instead of an unhittable target escaping as a setup error.

### What is reachable

`min_dist`/`max_dist` are the range of the *question*, not a promise that the answer exists —
but with `segments = 0` the promise holds over everything worth asking: every one of the 1224
distances from 0.01 m to 612 m, stepped by 0.5 m, was hit, the worst landing error 0.027 m against
`DIST_ACCURACY = 0.05 m`. Force a single crossing and refusals appear: the falling crossing
refuses below **7.4 m** (`the falling crossing got no closer to it than 7.4 m at 0.0068 rad`),
the rising one refuses above it. The model's own ceiling from level ground is 617.73 m at 0.7084
rad and **`max_dist` is that number** — it is derived from the model, no longer written down, so
the arrow you configure moves it. Between 612 m and that ceiling the range-angle curve is at its
flattest and the angle search can no longer verify a landing inside `DIST_ACCURACY`: those
questions are answered by a refusal from the search, which says which crossing failed, rather than
by a limit that quietly hides them. For a target above the archer the auto
search offers the practical shot and the plunging one is one attribute away: for a target 50 m up
at 120 m the model answers 0.456972 rad rising and 1.468337 rad falling — the same distance, two
different shots.

---

## Worked numbers

**Wind**, 100 m target, level:

| wind (m/s) | launch angle | impact speed | time of flight |
|---|---|---|---|
| -20 (head) | 0.050747 rad | 88.99 m/s | 1.043 s |
| -10 | 0.050011 rad | 91.28 m/s | 1.030 s |
| 0 | 0.049356 rad | 93.37 m/s | 1.018 s |
| +10 (tail) | 0.048773 rad | 95.25 m/s | 1.008 s |
| +20 | 0.048256 rad | 96.93 m/s | 0.999 s |

The last row's impact speed sits on top of the still-air terminal speed (96.93 m/s) — a
coincidence of arithmetic, worth spelling out because it is misleading: ground speed is air-frame
speed plus the 20 m/s of air, and in the air frame the arrow slowed from 83.63 m/s to 76.96 m/s,
which is what drag actually did to it.

**Altitude**, 200 m target:

| `archer_alt` (m) | launch angle | trajectory height |
|---|---|---|
| -100 (archer below) | 1.389756 rad (79.6 deg) | 253.63 m |
| -10 | 0.157081 rad | 2.01 m |
| 0 | 0.106473 rad | 5.71 m |
| +10 | 0.055920 rad | 11.62 m |
| +100 (archer above) | -0.361083 rad | 99.97 m |

Shooting uphill at 200 m is not a flatter shot: the flattest way to reach a target 100 m above
you at 200 m is a 79.6 degree lob, and the same target 100 m *below* you is hit by aiming 20.7
degrees **down**. The eye line is not the answer, in either direction.

**Aiming errors.** `get_vertical_distance` is the "how bad is it" function. At 50 m, around the
correct 0.024161 rad:

| launch angle | vertical error at 50 m |
|---|---|
| 0.02216 rad | -0.100 m |
| 0.02316 rad | -0.050 m |
| 0.02416 rad | 0.000 m |
| 0.02516 rad | +0.050 m |
| 0.02616 rad | +0.100 m |

One milliradian of aim is 5 cm at 50 m. The arrow keeps flying after it passes the sight line,
so a too-flat aim still gets an answer at any `x` in range: `get_vertical_distance(0.005, 300.0)`
returns **-49.81 m**, i.e. the arrow is 49.8 m below the sight line when it crosses 300 m, not an
error. Only when no point of the flight is at that `x` at all does the search fail — a headwind
strong enough to carry the arrow backwards is the way to see it (`wind = -60`, `a = 0.3`, asking
for `x = 400` gives `hit_y = 0` and `hit_x = 288.4 m`, the end of the flight) — and then it
reports where the flight ended instead of integrating toward it forever.

---

## What the model does not do

Honest limits of the equations being solved:

- **Two dimensions, no crosswind.** A crosswind would need the drag to be computed from a
  3-vector relative velocity; `wind` here is along the line of fire only.
- **No arrow dynamics.** No spin, no oscillation, no weathercocking, no fishtailing, no
  planing, no Mach effects. A real arrow is not a point mass; near its own resonance a fletched
  arrow's drag coefficient moves around.
- **Constant atmosphere.** `air_dens`, `gravity_accel` and `drag_coef` are constants; you can
  set them, but the model will not look them up, and drag does not change with speed regime.
  `k` is a single number, so the same `Cd` applies at 103 m/s and at 60 m/s.
- **Steady wind only.** Gusts, shear and a wind that varies with height are outside the
  Galilean-shift trick and would need a time-marched solution.
- **Flat target line.** The sight line is the horizontal plane through the archer's eye
  (`y = 0`), with the arrow released `ar_launch_alt` below it. No target size, no aiming
  point offset, no target face geometry: an "x" is a point.
- **Range validity is your data's range.** `max_dist` follows the arrow you configure — 617.7 m
  for the default MAG340, 1091 m with a slicker `drag_coef` — which is a statement about this
  model, not that `Cd = 0.4` is right at 600 m. Calibrate before trusting
  anything at the far end, and treat the numbers as a good model rather than a measurement —
  the spread between individual arrows and across a string's serving is larger than the
  numerical error.

---

## The numerical contract

Every search is bounded by a named constant, so no question can make it spin.
Timed on this host (CPython 3.14.4, median of 200 calls, one machine, not a property of the
model): an auto `get_angle` costs **1.6 ms** at 50 m and **2.1 ms** at 300 m on level ground; a
target 50 m above the archer at 120 m costs **47 ms**, because auto solves both crossings there and
the long one is a 16-second flight; the lofted route, **42 ms** warm and 76 ms cold. The whole
command, interpreter start-up included, is **53 ms**.

| constant | value | what it bounds |
|---|---|---|
| `_NODES`, `_WEIGHTS` | 8 | Gauss-Legendre nodes per panel, fixed |
| `PANEL_U` | 0.70 | max width of one quadrature panel in `u` |
| `MAX_PANELS` | 64 | max panels per quadrature |
| `U_LIMIT` | 22.0 | `|tan(theta)| ~ 2e9`, past any real arrow |
| `U_STEP` | 1.00 | step of the bracket walk in `u` |
| `BISECT_STEPS` | 16 | bisections per root search |
| `NEWTON_STEPS` | 5 | Newton polish steps (quadratic: 5 is plenty) |
| `ANGLE_MARCH` | 0.05 | growth of one launch-angle probe |
| `ANGLE_FLOOR` | 0.01 rad | smallest probe (a point-blank shot is ~0.007 rad) |
| `MAX_MARCH` | 40 | launch-angle probes per march |
| `ANGLE_STEPS` | 48 | steps of one branch refinement |
| `ANGLE_EPS` | 1e-12 rad | bracket collapse |
| `DIST_ACCURACY` | 0.05 m | how close a returned angle must really land |

The inverse problem is genuinely awkward, and the code is shaped around that. `distance(angle)`
is not a friendly curve: below some angle the arrow never gets back to the sight line at all,
then the distance grows to a maximum near 40 degrees and falls to zero at vertical — and where
the branch appears it **jumps**, so a solver that simply hunts for `distance == target` can stop
on a jump and report a hit its own forward model denies. `get_angle` therefore searches two
monotone branches in turn, then bisects for the jump itself and root-finds on the smooth branch
starting immediately above it (the hitting angles can hide in a window only a few thousandths of
a radian wide, which a march of any step size can step straight over), and returns an angle only
after the forward model agrees with it to `DIST_ACCURACY`. It returns the **flattest** hit: at
300 m there are two, 0.174606 rad and 1.307970 rad (74.9 deg), and it gives you the one you want
to take.

---

## Checks you can run yourself

Nothing here needs a test runner or a network; each line is a claim you can watch fail.

```
# round trip: the angle it gives really lands there
$ python3 arrow.py d=300
distance 300.0 m -> angle 0.17461 rad
$ python3 arrow.py a=0.174606
angle 0.174606 rad -> distance 300.000 m

# vacuum limit: with no drag the model must equal v0^2 sin(2a)/g
$ python3 -c "
import math, arrow
a = arrow.Arrow(); a.drag_coef = 0.0; a.ar_launch_alt = 0.0
for ang in (0.02, 0.05, 0.2, 0.5):
    d = a.get_horizontal_distance(ang)
    t = 2 * a.ar_launch_velo * math.sin(ang) / a.gravity_accel
    print(d - a.ar_launch_velo * math.cos(ang) * t)
"
7.105427357601002e-15
1.4210854715202004e-14
5.684341886080802e-14
3.410605131648481e-13
```

Two more worth knowing: `arrow.Arrow().terminal_speed()` must equal
`sqrt(g / drag_k())` (both 96.927 m/s by default), and `Arrow().max_range()` must equal the
largest value you find by scanning `get_horizontal_distance` over `(0, pi/2)` — brute force
gives 617.732 m at 0.7084 rad, and `max_range()` gives 617.732 m.

### The test suite

`arrow_test.py` sits next to `arrow.py` and needs nothing but the standard library:

```
$ python3 arrow_test.py
Ran 44 tests in 10.6s

OK
```

It is not a mirror of the source. It checks the public surface against a TABLE of angles and
distances; checks the forward model against an **independent RK4 march** written from the
equations in the test file (drag from the relative velocity, so it cannot share the closed form's
Galilean-shift trick), to a millimetre; checks the vacuum limit to nanometres; checks that nothing
falls faster than `sqrt(g/k)`; checks that the auto crossing search answers every distance a
forced crossing can, and hands back the crossing that is in the air for less time; checks that a
forced crossing which cannot reach falls back to the other and says so in `note`, and that
the note quotes the flatter alternative when there is one; checks that a forward call follows the
crossing that answered; checks the wall recipe — that the plunging shot clears at half range what
the flat shot goes under; checks that an angle past the vertical guard is a refusal and not a
setup error; checks that `calibrate` turns a chronograph reading back into the drag that made it;
and checks the two interfaces: that the command line applies its settings before the question in
any order and prints the note where the angle is read, that the REPL takes `d=50` and `d50` alike,
that it answers in the order it is asked in, that its `h` prints the same screen as `-h`, and that
a nonsense line is refused without applying half of it. It checks the lofted route: that
`steepest_angle` returns an angle the forward model agrees with at targets from 1 m to 612 m, that
`segments = 1` and `2` cannot reach it, that `segments = 3` reaches it through `get_angle` and the
command line alike, that the note prices it against the flat route, and that the refusal below
0.12 m says why there is nothing to return. And it checks the three promises the file makes about
itself: standard library only, no recursion in the call graph, and no loop a question can spin.

The tests were shown to catch what they claim. On a throwaway copy of the module, asking the
falling crossing first, refusing instead of falling back, never writing the note, ignoring
`crossing` in the forward calls, and letting the vertical guard escape each turned the suite red,
naming the test that should name it. Eight of nine mutations of the new code were caught the same
way: searching the flat side of the peak instead of the steep one, dropping the crossing pin,
dropping the too-close refusal, silencing the lofted note, refusing `segments = 3` in the model and
in the interface, importing something outside the standard library, and putting an unbounded loop
in the physics. The ninth was not: deleting the final `DIST_ACCURACY` re-check of the angle
`steepest_angle` returns. Nothing in the code as it stands can make that check fail — the bisection
is accurate on a monotone bracket — so it is an equivalent mutant rather than a bug, and the check
is there for the day the bracket stops being monotone. Saying so here is cheaper than a test that
could only ever pass.

### The numbers in the tables

Tables and prose are not code blocks, so the block checker above skips them — and they carry the
numbers a reader actually quotes. `test/check_crossings.py` recomputes every one of them in
[Which crossing the arrow takes](#which-crossing-the-arrow-takes): the level-target table, the
7.399 m hand-over, the flatter-and-faster claim over all 120 geometries, all 243 answers of the
nine-altitudes-by-nine-distances-by-three-settings sweep against the independent RK4 march, and
the lofted twin — found by bisection on the forward model at every distance from 1 m to 612 m, so
that the numbers in the section above are recomputed rather than remembered. About a minute, exit 0 means
`0 claims broken - the README's crossing section holds`:

    python3 test/check_crossings.py

The source passes `ruff check` with no findings (ruff 0.17.0, default rule set), `arrow.py` and
`arrow_test.py` both. The three promises about the file itself — imports are `math`, `sys`,
`typing` and nothing else; every search loop is `for _ in range(<named bound>)`; no function in
the module can reach itself — are checked by `arrow_test.py`, since they are promises about the
code rather than about archery.

---

## Version

**3.1.0** — the other shot at the target:

- **`steepest_angle(distance)`, and `segments = 3`, give the lofted route.** Anything hittable can
  be hit flat or lofted, and `get_angle` only ever returned the flat one; for a target level with
  the archer the lofted twin is on the *same* crossing as the flat one, so `segments` could not
  reach it either. It walks down the branch from the vertical end instead — same closed form, one
  bracketed bisection, every returned angle verified by the model's own forward call — and writes
  `note` with what the lofted shot costs beside the flat route's angle and time. Below about
  0.12 m there is nothing to find and the refusal says so. See
  [Shooting over a wall](#shooting-over-a-wall).
- **`max_dist` is derived from the model, not written down.** It was `612.`, a number noted for the
  default MAG340 arrow, and it became a lie the moment anyone changed the arrow: a slicker
  `drag_coef` carries 1091 m in this model and the constant refused 700 m it could hit with room to
  spare — reachable through `calibrate()`, a documented feature. The limit is `max_range()` itself
  now (617.7 m by default, 666 m with an 8 m/s tailwind, 1091 m for the slick arrow), it stops
  caring which crossing `segments` is asking for, and it is remembered against the twelve settings
  that move a trajectory — so a question costs nothing extra (1.6 ms before and after; deriving
  costs 16 ms and happens once per setup, not twice per question, which the read-back's own writes
  would otherwise cause). Refusals name both numbers now: `700 m is past the limit 617.7 m (this
  arrow carries at most 617.7 m at these settings …)` instead of `target_distance=700.0`. Where the
  model cannot bring an arrow back to the sight line at all, the limit is `inf` rather than the
  search's `-1.` marker, so the search keeps the sentence that explains the setup.
- **AGPL-3.0-or-later**, canonical text in `LICENSE`, SPDX line in both Python files.
- **The read-back is renamed, because `final_*` was lying.** `final_x` and `final_t` were filled at
  whatever point the call had been asked about — asking how high the arrow is at 120 m on a shot
  that lands at 574 m left `final_x` at 120 — so the family is now `hit_x`, `hit_y`, `hit_speed`,
  `hit_energy`, `hit_time` and `hit_angle`, with **`crossing`** for the answer to `segments`,
  **`apex`** for the top of the arc and **`note`** for what the solver said about its own choice.
  `hit_angle` is the angle the arrow arrives at; the launch angle stays an argument, never an
  attribute. The answer block is printed as a table, labels in one column and the `=` in one.
- **`final_h` is gone.** This is the first published version of Arrow, so there is nothing to stay
  compatible with: the highest point of the arc has one name, `apex`, and a test fails if a second
  name for it ever comes back.
- **The README stopped talking to developers**: function and loop counts are gone from it, and the
  three promises they carried are now checked by the test suite instead.

**3.0.2** — behaviour and messages, not physics:

- **`segments` defaults to auto, and a crossing that cannot reach is no longer a dead end.** `0`
  asks the rising crossing, then the falling one, and `final_s` reports the winner; `1` and `2`
  still force one, but when the crossing you asked for cannot hit the target you get the other
  crossing's answer plus a `final_note` saying what you asked for, why it cannot, and (when both
  exist) the flatter route you are not taking. `Unreachable` now comes only when neither crossing
  can hit. This closed the 0.01–7.4 m hole where a level target could not be hit at all. See
  [Which crossing the arrow takes](#which-crossing-the-arrow-takes).
- **The two interfaces speak one grammar, and say so.** `-h` / `--help` (and the REPL's `h`) print
  the whole interface on one screen. The `=` is now optional in the REPL — `d50` and `d=50` are the
  same line, and always should have been: the REPL used to forbid the `=` the command line required,
  so typing the familiar form produced `could not convert string to float: '=50'`. One parser, one
  settings application, both interfaces, and a nonsense token now names itself.
- **The REPL answers in the order it is asked**: what you typed, then what the arrow is set to, then
  the answer, then the numbers of that shot, with `final_s` first among them because it says which
  shot the rest describe. Settings used to be printed after the answer they produced.
- **`max_h` replaces `final_h`**: the highest point of the arc is not a quantity at the end of the
  flight, and calling it `final_*` invited exactly the wrong reading. `final_h` survived as a
  read-only property until 3.1.0 removed it.
- **A forward call now follows the crossing that answered.** `get_horizontal_distance` and
  `get_vertical_distance` with `segments = 0` use `final_s`, so re-checking the angle `get_angle`
  just returned measures that shot instead of the other half of the arc. With no shot asked for
  yet they use the falling crossing, as before. Setting `segments` explicitly still means it.
- **A search probe past the vertical guard is a refusal, not a setup error.** Asking for a target
  200 m above the archer at 1 cm used to raise `BadInput: launch angle too close to vertical`,
  which reads as a broken bow; that crossing now refuses with the guard quoted, and `BadInput` is
  reserved for what the caller actually got wrong — including a wind at or beyond the launch speed,
  which used to be discovered only when a probe stumbled into it.
- **Command-line settings are applied before the question**, whatever order they are typed in.
  They used to be read left to right, so `main()`'s own documented example, `d=50 l=10 s=2 w=3`,
  silently answered the level shot and ignored `l=`, `s=` and `w=`. The regression is a test.
- **Refusals say what was tried** — the closest each crossing got, or that it never reached the
  sight line — instead of the search's internal `-1.0` marker leaking out as a distance.
- **The REPL banner names the file** (`arrow.py(3.0.2)`), and the shebang is `#!/usr/bin/env
  python3`, so `./arrow.py` works on a host with no plain `python`.

**3.0.1** — the closed form: the first integral, the `u = asinh(tan theta)` coordinate, the fixed
Gauss-Legendre quadrature, and every loop bounded.

**2.x** — the same physics, marched: a fixed Euler loop over flight time, with the launch angle
found by stepping through angles until the distance came out right. Same model and same answers,
about **15× slower** than the closed form on this host. The factor is the author's own timing of
the 2.x code, which is not in this repository and so is reported here rather than recomputed; the
3.0 rewrite replaced the march with the first integral and nothing of 2.x survives in the file.

---

## Files, testing, license

```
README.md
arrow.py
arrow_test.py
LICENSE
```

One module, no package, no dependencies, no build. `arrow.py` is self-contained: nothing outside
the file is needed to import it, run it or trust it; `arrow_test.py` needs only it and the
standard library. `VERSION` in the file is the version of the model and solver (`3.1.0`); the
header docstring is the derivation, and every claim in it is one you can check with the commands
above.

**License: AGPL-3.0-or-later.** The canonical text is in `LICENSE`; `arrow.py` and `arrow_test.py`
each carry `SPDX-License-Identifier: AGPL-3.0-or-later`, and both are Copyright (c) Jan Hercek. Use
it, change it, put it in a product — the one obligation beyond the ordinary free licences is the
network clause: if you run a modified version as a service other people use, you have to offer
them its source.
