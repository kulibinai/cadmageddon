# Reconstructing a FreeCAD feature tree from a mesh

You are given a mesh and a set of measuring instruments. You have no ground-truth
code and no dimensions. Measure the part with the instruments, then build it as
a FreeCAD program: a PartDesign feature tree, one feature per `sketch` call
(Pad, Pocket, Revolution, Groove, Loft, Pipe, Helix, patterns). You never write
that code: `sketch` writes it from the names you choose, and saves the tree as
an .FCStd beside the program.

This document explains how to think and in what order to work. For the exact
commands and every flag, read `instruments.md`. It is generated from the CLI
itself, so it is always current, and it also lists any instruments added after
this document was written.

**A note on the word "sweep".** In tool output, *sweep* usually means the scan
the tools run to find levels: the part is cut by many parallel planes along a
plane's normal, and the cross-sections are compared ("sweep step", "sweep only",
"sweep resolution", `--levels sweep`). This document calls that **the scan**.
It has nothing to do with `--op sweep`, which is the operation that carries a
profile along a path (a FreeCAD Pipe or Helix).

## The loop

    overview -> survey -> spectrum -> ortho -> section -> sketch --check
                                                 ^               |
                                                 +-- residual <--+

Build one feature at a time. After each `sketch --check`, the last line is the
**residual**: the volume still missing and the volume that is extra, in mm^3.
It is the only reliable signal that you are finished.

    check: residual   MISSING 4109.0 mm^3 in 2 bodies ...

**Use the band table as your plan.** Under the level table, `section` prints
one row per band, i.e. per pair of adjacent levels: its depth, loop count, hole
count and area. A band with a constant section is usually one feature, so a
single call tells you how many features the part has and where each starts and
stops. Discovering the same thing by sectioning at arbitrary offsets costs one
call per cut and can use up most of your budget before you build anything.
`faces` and one section through the body are usually enough to build the first
feature; `--check` then says what is left. Runs that spend a third or more of
their calls looking before the first build tend to run out of budget on the
features they build last.

**Read the notes under the level table.** They list what the scan may have
missed. The most important is the *blind radius*: a taper (a chamfer or a draft
angle) that ends within that distance of a step or of the part's end produces no
level at all. If you model that region as straight prisms, the taper is simply
lost. When such a note suggests a specific `--samples` value, re-run the same
command with that value. It costs one call and usually reveals the missing
level.

**Treat a leftover body as a part in its own right.** `residual --export
left.stl` writes whatever is still missing to a mesh, and every instrument works
on it. Run `overview`, `survey` and `section` on `left.stl` to measure the
missing feature directly instead of inferring it from the whole part. This is
usually faster than reasoning about what the difference between two solids must
look like, and it works well for awkward shapes such as a blend between two
features.

**You are not finished while the residual is non-empty.** A part that looks
right in the renders but still has a missing body has a feature left out —
often a thin one, such as a chamfered foot under a box. When the residual names
bodies, run `residual` to see where they are, then build them. Each side of
the residual, and each body `residual` lists, names the target's faces it lies
on (`on cone F12 + plane F3`): the `F#` that `faces` describes and `--to F#`
takes. Read where a leftover is from those faces, not from its centre.
`residual` also places each body by the named levels around it and the ring
there (`at Z B1..E1 R0`: section at `mid(B1,E1)`, use `R0`), and lists the
target faces the build misses with which way: "built short of it by 0.4 mm"
is a face to move out, "built past it" one to bring in. The
reconstruction is done only when the residual reads "nothing above the sliver
floor either way". In the last stretch `--check` also weighs the target's
surface (`check: surface`), each patch named by its faces and its side: shaped
patches there are a shape the volume cannot see -- a bend built sharp, a draft
built straight -- and DONE is refused once while they cover more than 1% of it.

**Cut your program where the target is wrong.** `section <mesh> --plane P
--offset <a level through the leftover> --built PROG --fill` sections your
program at the same place as the target and paints the difference: green
where the target has material you have not built, red where you built material
it does not have. Each region is listed (`M1`, `X1`, ...) with its area, centre
and the target ring it lies on -- a red disc on a hole ring is a hole you did not
cut. Name that ring in the next `sketch`.

**Fix a wrong feature in place instead of building around it.** Every `--check`
ends with a `program:` line that numbers the features (`#0 extrude XY ... | #1
loft XY ...`). When one of them is wrong (the wrong profile or level, or a
check that reports pieces or NOT A VALID SOLID), replace it with
`sketch ... --base PROG --replace 1 --emit PROG --check`, remove it with
`edit <mesh> --built PROG --drop 1 --emit PROG --check`, or go back to the
program before your last `--emit` with `edit <mesh> --built PROG --undo --emit
PROG --check`. The `vs last saved version` line under the residual says whether
the change helped; if it got worse, `--undo` it.

## Order of work: largest features first

Build in order of decreasing size. Establish the main body first (the prism or
revolve that accounts for most of the volume), then add bosses, holes and
pockets, and finally chamfers and fillets.

- **Every later feature is positioned relative to the main body.** An error in
  the main body is inherited by every later coordinate, and it is hard to find
  because the part still looks broadly right.
- **The largest feature is the best-measured one.** It spans the most scan
  samples and the longest section edges, so its dimensions have the smallest
  relative error.

**Cut holes and pockets that run across the main body; don't build around
them.** A hole that shows as an inner ring of the main body's section comes with
its extrusion. One that runs another way -- a cross-hole, a slot, a counterbore
from a side face, a spherical seat -- is its own feature: section where it shows
as a ring and `sketch ... --use <that ring> --combine cut --base PROG`. A ring
named alone is filled and subtracted, and any `--op` builds the cutter (a
revolve for a stepped bore or a sphere).

A drilled hole with steps -- a counterbore, a countersink, chamfers, a drill
point -- is already measured: `faces` lists it under `holes` as `C#`, and
`sketch <mesh> --op hole --use C1,C2 --base PROG --emit PROG --check` cuts it
with every step, as many holes as you name.

**A rounded or bevelled edge is one command once the body is built sharp.**
A chamfered perimeter, 45 deg chamfers at an end, a fillet where a wall meets a
flange, the bend of a sheet: build the body with square corners -- a loft or a
profile redrawn to approximate them costs calls and leaves wedges -- then
`sketch <mesh> --op fillet --use F12,F13 --base PROG --emit PROG --check` (or
`--op chamfer`). `faces` names them under `blends`, and `--check` and
`residual` name them again (`blends`) when a leftover body lies along one. Cut
on a convex edge, added into a concave corner; name several at once, all
convex or all concave. A face's own outline is not square where they have cut
its edge back: `sketch --face F#` says so when its outline meets faces that
lean instead of walls, and `--face F# --sharp` carries the walls on to their
corners -- the sharp block to cut them from. A block extruded from the cut-back
outline is short at every level, and no loft adds back what a chamfer removes.

**A face or a hole that lies across no axis is built in its own frame.**
`faces` lists such directions under `directions`, with the faces on each. `sketch
--op hole --use C#` on a stack along one, and `sketch --face F#` on a flat face
of one (its outline, extruded `--to` a level across it, such as the parallel
face), set the frame themselves. For anything else along a tilted direction -- a
boss, a profile across it -- give the same `--frame F#` to `ortho`, `section`
and `sketch`: levels and rings are then the ones across that direction, and the
feature is turned back when it is written.

## Prisms and revolves need different plans

`overview` prints a `revolve` line with an axis and a score. A score above about
0.93 means the part is close to a solid of revolution about that axis, and the
prism workflow will not help: slicing perpendicular to the axis of a revolved
part gives a cross-section that changes at every step, so `survey` reports
near-zero prismaticity and `spectrum` finds no plateaus. The part is fine; the
approach is wrong for it.

For a revolved part, section on a plane that **contains** the axis rather than
one perpendicular to it. The cut is then the profile that was revolved, and you
build it with `--op revolve`:

- Axis Z: section on XZ or YZ. Axis Y: XY or YZ. Axis X: XY or XZ.
- A sphere is a HALF circle revolved about its own diameter, so name one half:
  `--use "A0_0@upper" --op revolve`. Each half-arc option prints the diameter it
  was split on, and that line is the revolve axis.
- Take the section through the axis, not off to one side. An off-axis cut gives
  a chord through the solid rather than its true profile.
- When the cut through the axis also crosses holes, blades or teeth, it is not
  the profile. `section --meridian F#` (a round face or hole stack on that axis)
  cuts at every angle and gives the envelope (material at any angle: holes
  filled) and the core (material at every angle: blades and teeth gone);
  `sketch --meridian F# --use R0 --op revolve` builds one, then cut or add what
  is not round. `--split "Z=0" --keep +` keeps one side of both profiles, as on
  a plane section: the boss above a gear without the teeth's full ring.

**A drafted wall is one extrude.** A pocket, socket entry or cast wall whose
outline shrinks or grows steadily between two levels is `sketch ... --op extrude
--to <level> --taper`: the draft is measured from the two profiles, never typed.
When it answers "not a draft" (the far profile is not the near one offset by a
constant -- corners that keep or grow their radius, say), `section ... --law
<level>` fits a circle, slot or rounded rectangle at levels in between and shows
whether its numbers follow straight lines; when they do, `sketch ... --op loft
--law` builds it from them, one ring at a time.

**A prism that is round at its ends is a prism clipped by a revolve.** Build
the prism, then the revolve (its meridian envelope, say) with `--base PROG
--combine intersect`: only material inside both is kept, so the ends come out
turned instead of stair-stepped. A joggled or bent plate is its flat outline
extruded and clipped by the strip a cut across the bend gives; no single cut
shows that outline, `--project <level>` does (everything between the two levels
laid flat). `--with N` clips feature #N alone and leaves the rest of the part
as it is.

Do not trust the `regime` label over the score: it has labelled a part with a
revolve score of 0.986 "patterned". Go by the score.

## Colour conventions

`overview` draws the target only, even when you pass `--built`.

Images that overlay the target and your reconstruction by colour channel use
this code:

    green   = target (what is still missing)
    red     = your reconstruction
    yellow  = both agree
    black   = empty space

Each solid is rendered separately, shaded by depth along the view direction,
and the two images are combined by colour channel. This cannot show a
difference hidden *behind* other geometry: it reads as agreement. That is why
there are several views, not one.

`residual` draws its own table instead: green = the MISSING bodies (`D#`), red =
the EXTRA bodies (`E#`), grey = what you have built, dim = a skin (a face built
off the target's: large area, little volume). Every body is drawn in front of
the build, so one hidden behind it still shows. `section --built --fill` uses
the same green and red, with grey where the two sections agree.

`progress` uses its own scheme: each solid is painted by its distance to the
other, and the image caption states what "slight" and "large" mean in mm.

## The instruments

All rendering commands accept `--tile N`, the size in pixels of each view or
pane. The defaults suit most parts. Raise it when small features or closely
spaced level lines are hard to read; a larger image costs more to read.

    m2c.py overview  <mesh> [--tile N]
        Ten projections: four isometric views at one elevation and four
        azimuths, plus the six axis-aligned views. Also prints the bounding box,
        volume, watertightness, the dominant flat-face normals, and the revolve
        score. Use it to orient yourself, not to read exact shapes: isometric
        views foreshorten. The `regime` label is one word for the whole part;
        the lines under `revolve` say over which stretch of the axis the part
        is round and where it is not, and the `prism` lines where along each
        axis the cross-section stays the same. A part round over most of its
        length with one stretch that is not is a revolve plus a feature there,
        whatever the whole-part score. A target exported face by face has
        cracks along its edges; ingest zips them and says so on every command
        ("note: the target's N boundary edges were cracks ..."). A target still
        "not watertight" after that has a real hole or a shared edge: sections
        and band tables near it can be wrong, so check them against the views.

    m2c.py survey    <mesh>
        Scores XY, YZ and XZ by how much of the part's extent has a constant
        cross-section, and recommends one. The recommendation maximises
        prismaticity, which is not always the plane a designer would have
        sketched on. Override it when another plane gives simpler features, and
        state that you did.

    m2c.py spectrum  <mesh> --plane XY|YZ|XZ [--samples N]
        Scans the part along the plane's normal. Returns runs of constant
        cross-section, each with the extrusion depth it implies, the levels
        where the cross-section changes, and an area profile.
        `loops` and `holes` are separate columns: a disc with nine holes through
        it is still ONE loop.
        Do not use a run's `depth` as an extrusion length. The scan starts and
        stops slightly inside the part, so the depth reads short. Measure from a
        real face to a level instead.

    m2c.py ortho     <mesh> --plane XY|YZ|XZ [--offset LEVEL] [--samples N]
                     [--levels both|faces|sweep] [--no-boundaries] [--tile N]
        The four elevations in which the plane is seen edge-on, so each level
        appears as a horizontal line, with a scale along the normal axis. Omit
        `--offset` while you are still choosing where to cut.

        Levels are read from the part's FACES, and each is labelled by kind:

        - `E`, a PART END: the bounding box, so it is exact. Always drawn in
          bold, with arrowheads and a filled label.
        - `B`, a flat face across the axis ("flat face F18, facing -Z,
          material above it"): where an extrusion starts or stops. Exact. A
          feature built from this level goes toward the side the material is
          on.
        - `K`, where a cone, a fillet, a sphere band, a sloped face or a
          revolve starts or stops. The section changes between two `K`s (or a
          `K` and an end) and is constant outside them: that span is a loft, a
          chamfer or a turned profile.
        - `T`, the widest ring of a torus, a sphere or a revolve, or the
          narrowest ring of a revolve whose profile waists in.
        - A curved surface whose axis lies ACROSS the scan direction, such as a
          cylinder lying on its side, produces no level; the notes name those
          surfaces. Build such a feature along its own axis, sketching on the
          plane in which it is round.
        - A level labelled "sweep only" is where the cut area changes but no
          face ends there. It is an estimate, not a measured face.

        `m2c.py faces` lists the faces themselves, and `section --face F18`
        takes a profile directly from a flat face. A face of kind `revolve` is
        a spline turned round an axis: `faces` gives the axis and the profile
        as (place along the axis, R), ring by ring. Build it as a revolve:
        section on a plane containing that axis, `--op revolve` the half
        profile, then check the radii against those rings.

        `--no-boundaries` hides the level lines and keeps only the part ends.
        Use it ONLY when the lines are so dense that they hide the geometry,
        typically on a rounded or tapered part with many of them. On a part with
        one or two levels, the lines are the most useful thing in the picture.
        Hidden levels are still listed in the text output.

    m2c.py section   <mesh> --plane XY|YZ|XZ --offset LEVEL
                     [--side above|below] [--face F] [--split LINE --keep +|-]
                     [--tol MM] [--grid MM] [--tile N]
        The cut, decomposed into named LINES and ARCS with fitted radii,
        centres, angular spans and fit residuals, plus any hole pattern. Also
        re-renders the four elevations with the cut level marked.
        A requested level snaps to the nearest face or level and is kept as the
        SKETCH PLANE, but the cut itself is taken slightly inside the solid,
        because a cut exactly on a face is degenerate.
        At a level where TWO different prisms meet, both sections are shown,
        numbered continuously, unless you choose one with `--side`. How the
        smaller one sits inside the larger is usually the point.
        Each primitive's line carries its two end points. A ring of eight or
        more primitives opens with a `shape:` line, its line lengths and arc
        radii grouped by size with counts ("arcs R2.74-3.10 x14"), so a ring
        of repeated teeth or fillets reads as what repeats before the list.
        A ring of 24 or more prints its repeats as groups and only its 40
        longest other primitives; `--detail R#` lists every one of that ring.

    m2c.py sketch    <mesh> --plane XY|YZ|XZ --at LEVEL --use NAMES
                     [--op extrude|revolve|loft|sweep] [--to LEVEL] [--angle DEG]
                     [--base PROG] [--emit FILE] [--check]
        Composes a sketch from primitives named by `section` and emits one
        FreeCAD feature (a block of the feature tree).

            --use "A0_0,A1_0"     an outer ring and a hole ring
            --use "L0_0,L1_0"     two separate OUTER rings, unioned; this is
                                  what a tube's wall looks like when cut in the
                                  plane of its own bend
            --use "A0_0@upper"    the upper half of that circle
            --use "A1_0@right"    the right half

        Rings that came from a hole are subtracted automatically. Halves are
        named `@upper`, `@lower`, `@left`, `@right`, or by axis, `@Z+`, `@X-`.
        At a level between two prisms, the side (and with it the extrusion
        direction) is inferred from the names you use. `--to` takes a level
        NAME, `mid(A,B)` or a number; there is no `boundary` or `end` keyword.
        `--base PROG` unions the new feature onto what you built before
        (`--combine cut` subtracts it instead). A part made of several
        bodies -- `overview` says how many (`bodies`) -- gets each one with
        `--combine separate`: kept a body of its own, not fused into the rest.
        `--check` says when one body you built fills two of the target's.
        `--check` builds the result and prints its bounding box, volume and
        residual.

        **Only `--emit` writes the file. `--check` alone measures a program
        that is never saved.** It can report the part as finished while the
        file on disk still holds whatever your last `--emit` wrote. When a check
        comes back clean, re-run the same command with `--emit` before you say
        DONE.

        Naming one primitive selects the WHOLE ring it belongs to. A rectangle
        is `L0_0`; do not list `L0_0,L0_1,L0_2,L0_3`. Name one primitive per
        ring.

#### The four operations, and when to use each

        --op extrude   the section is CONSTANT between the two levels. The band
                       table shows this: one band, one loop count, one area.
        --op revolve   a body of revolution. Section on a plane CONTAINING the
                       axis and revolve the half profile; `--split` cuts a full
                       section into halves first.
        --op loft      the section CHANGES between the two levels: a taper, a
                       draft, a chamfered cap. Extruding the mid-section instead
                       overfills (a tapered spoke by over 15%). Add `--smooth`
                       when the section rotates or bends between the levels
                       rather than just shrinking; a helical gear tooth built
                       with the default straight blend loses 2.5 IoU points.
                       `--to B1,B2,B3` lofts through several levels in turn.
        --op sweep     a profile carried along a fitted path: `--along H0`,
                       the helix `overview` reports on a coil or a thread, or
                       `--along W#`, the centreline `section` names inside a
                       strip ring R# -- a wire or pipe bent in one plane, cut
                       in that plane; a round wire of the strip's width is
                       swept along it. Along H0, any profile, not only a round
                       wire: cut the section through the axis at the core's
                       radius (`--split`), name one whole tooth (`--profile
                       R#`, as `section --split` lists them), and give the
                       stretch of the axis it covers (`--between K0,K1`). A part with
                       a worm or screw beside other bodies gets one helix per
                       body (H0, H1 ...), each with the axis it is wound on
                       and the stretch it covers; `overview` of a `residual`
                       export fits the leftover thread on its own.

#### Modifiers worth knowing

        --both         extrude the same distance on both sides of the sketch
                       plane. Without it, a feature centred on the level where
                       you measured it (a spoke cut at its mid-thickness) has to
                       be built as two halves.
        --pattern      "5@Y=0,Z=0" repeats the NEW feature five times, evenly
                       spaced about that world axis. Build one spoke and a
                       pattern, not five spokes. "4@B1..B4" repeats it in a
                       row, the last copy moved by the distance from B1 to B4
                       ("4@X:B1..B4" along another axis).
        --mirror       adds the feature's mirror image: about S0 from
                       `symmetry`, or a level such as M0 or mid(E0,E1)
                       ("Y:M0" on another axis). Build one side of a
                       symmetric part and mirror it. `edit --feature N
                       --mirror S0` (or `--pattern ...`) does the same to a
                       feature already built, a cut too.
        --grow         pushes named profile edges outward, so a feature
                       overlaps what it joins instead of only touching it.
        --face F       sketches directly on a flat face named by `faces`.
        --frame F      works across an inclined face or axis (`faces` lists
                       them under directions); the same flag on ortho, section
                       and sketch.

        These are used far less often than the geometry calls for. If you find
        yourself building the same shape twice at different angles, you want
        `--pattern`.

    m2c.py progress  <mesh> --built PROG [--views iso|ortho] [--plane P]
        Target and reconstruction side by side, each painted by its distance
        to the other.

    m2c.py residual  <mesh> --built PROG [--plane P] [--export FILE] [--solo]
        The boolean difference between target and reconstruction, split into
        named bodies with volume, centre and (with `--axis`) angle: your
        remaining work list. `--plane` draws the elevations with each body's
        levels marked, ready to pass to `--at` and `--to`.

    m2c.py edit      <mesh> --built PROG [--drop N | --undo] [--emit PROG] [--check]
        Without --drop or --undo: lists the program's features (#0, #1, ...)
        with each one's own volume and bbox. --drop removes one; --undo
        restores the version before the last --emit. `sketch --replace N`
        swaps one for a new feature.

    m2c.py faces     <mesh> [--cover F# [--between A,B]]
        The planes, cylinders, cones, spheres, tori and revolved splines the
        mesh was built from, named (F0, F1, ...), with how each pair meets.
        `--cover F#` checks a reading before you build on it: a cylinder
        continued over the stretch (or the whole part) and how much of it the
        target really has -- a bore you think runs through may be a groove.

    m2c.py symmetry  <mesh>
        Mirror planes and rotation axes, each with the `--split` or `--axis`
        value to pass.

### Revolving a general profile

    section ... --split "X=0" --keep +|- [--use NAMES]
    sketch  ... --split "X=0" --keep +|- --axis "X=0,Y=0" --op revolve

A revolve profile is usually a mix of lines and arcs: a cup, a cone, a stepped
shaft. `--split` cuts the section along a line, keeps one side, and closes the
profile along the cut, so ANY section becomes a valid revolve profile. Run it
with `section` first and LOOK at the result: the split profile gets its own
names, and small stub artefacts the cut can leave are visible there and nowhere
else.

`--axis` gives the revolve axis in world coordinates. One of its two
constraints must fix the section plane's normal coordinate at the level you
sketched on; otherwise the axis does not lie in the profile's plane:

    revolve about Z, sketching on XZ at Y=0   ->  --axis "X=0,Y=0"
    revolve about Y, sketching on XY at Z=0   ->  --axis "X=0,Z=0"
    revolve about X, sketching on XY at Z=0   ->  --axis "Y=0,Z=0"

Name the axis instead of typing it whenever there is a name: a round face's
axis (`--axis F12`), a helix `overview` fitted (`--axis H0`), a rotation axis
from `symmetry` (`--axis S2`), or levels for the values (`--axis
"X=mid(E0,E1),Y=B0"`). `--split` takes the same: "X=M0", a mirror plane S0, a
flat face F#, or an axis lying in the plane (H0, a round face F#) for the line
along it.

### Tolerance

`--tol` is optional and normally omitted. By default the fit tolerance is
derived from the mesh's own sampling density, ring by ring. A fixed tolerance
is a guess about someone else's mesh: too loose, and a real 0.68 mm fillet
disappears and is replaced by arcs of R=25; too tight, and a straight wall is
fitted as an arc of R=407. Override it only for a specific reason, and state
the reason.

### A face carrying a chamfer is not the profile

If the level table shows two closely spaced levels at an end (say a part end at
0.000 and a `B` or `K` level at 1.794), the gap between them is a chamfer or a
fillet, and a section taken at the end face lands inside it. The measured
profile will be smaller than the prism's true profile, with nothing to warn
you. Take the measurement inside the plateau, but put the sketch plane on the
face. Those are two different numbers, and `section` reports both.

Do not retype coordinates that a command has already printed. Copying a radius
from `section` output by hand is where transcription errors creep in; naming
the primitive avoids them entirely.

## Reporting

State what each command told you and what you concluded from it. Whenever you
override a recommendation or assume something the tools did not tell you, say
so: an assumption that is written down is cheap to check, one that is hidden is
not.

## Planned: finding fillets and chamfers from the reconstruction's edges

Not yet implemented: today `sketch --op fillet|chamfer` builds the blends
`faces` names. Recorded here because the section-based approach has a hard
limit that this one does not.

A section only reveals a blend that happens to lie in the cutting plane. A
fillet along the profile of a revolved part shows up as an arc, but a chamfer
around a circular edge leaves no trace in a section perpendicular to that edge's
axis. It appears only as two closely spaced levels, which look like any other
pair of levels.

The edge-driven method would instead:

1. Reconstruct the part without any blends.
2. Sample points densely along the EDGES of that reconstruction.
3. Measure each sample's distance to the target mesh.
4. Where the distance exceeds the fit tolerance, the edge is blended in the
   target and sharp in the reconstruction.
5. Show a cut perpendicular to the edge at that point, with the target
   overlaid, and propose parameters: an arc of radius R for a fillet, or a
   straight bevel of leg L at angle A for a chamfer.

This works for any edge in any orientation, needs no guess about which plane to
cut on, and gives the parameters directly.
