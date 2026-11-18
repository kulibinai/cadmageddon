# CADMAGEDDON: High-Fidelity CAD Reconstruction with Local Inference

**CADMAGEDDON** ([arXiv:2611.22563](https://arxiv.org/abs/2611.22563)),
an open-source agent that reconstructs editable CadQuery programs from meshes and
runs on a locally served model.

A vision-language model (Qwen3.8-27B in the paper) makes the decisions — which
planes to cut, which features to build, what to correct next — by looking at
renders and reading the output of geometric instruments. The instruments do the
measuring and fitting: the model names entities ("the level B7", "ring R2") and
the harness turns them into exact parameters. The model never sees the mesh
file, the ground truth or the score.

## What is here

| Path | What it is |
|---|---|
| `agent_run.py` | The agent: runs the reconstruction loop on one mesh or on a directory of meshes |
| `evaluate.py` | Scores a directory of reconstructions: IoU, GMS (Generalized Match Score - introduced in [CADENA](https://arxiv.org/abs/2608.00799v1)), Chamfer distance, invalid rate |
| `mesh2cad/` | The harness library: normalisation, sections, fitting, program emission, rendering |
| `scripts/m2c.py` | The instrument CLI — the only commands the model may run |
| `scripts/export_stl.py` | Exports a delivered program to STL in the input mesh's frame |
| `scripts/pick_best.py` | Picks the version (and attempt) to deliver by smallest residual, without ground truth |
| `scripts/compare.py` | Voxel occupancy used by the IoU fallback; also a standalone visual comparison |
| `metrics/` | `pair_metrics.py` (IoU / CD / GMS of one pair) and `gms.py` (vendored from CADENA) |
| `prompts/` | The system prompt: `reconstruction.md` (method) and `instruments.md` (generated CLI reference) |
| `serving/serve_vllm.sh` | Reference vLLM launch with the settings used for the paper's runs |

## Install

Linux or macOS, Python 3.11, CadQuery/OpenCASCADE from conda-forge. A GPU is
needed only to serve the model.

```bash
conda env create -f environment.yml
conda activate cadmageddon
```

Rendering is offscreen (PyVista/VTK). On a headless Linux machine set
`PYVISTA_OFF_SCREEN=true` (the runner sets it for its own subprocesses) and,
if VTK has no EGL/OSMesa, run under `xvfb-run`.

## Serve the model

Any OpenAI-compatible endpoint works. The paper used vLLM with
`Qwen/Qwen3.8-27B-FP8`; `serving/serve_vllm.sh` is the launch we used, with
every setting overridable by an environment variable:

```bash
VLLM=$(which vllm) TP=4 PORT=8000 ./serving/serve_vllm.sh
```

The FP8 weights (~30 GB) fit on a single 48 GB GPU (`TP=1 GPUS=0`).

## Run

One mesh, or every mesh (`.stl`, `.obj`, `.ply`, `.off`) in a directory:

```bash
python agent_run.py --mesh data/meshes/part.stl --out runs/ \
    --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B

python agent_run.py --mesh data/meshes/ --out runs/ --jobs 8 \
    --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B
```

Each mesh gets `runs/<stem>/`:

```
part.stl            the input, as the agent sees it
run/                reconstruction.py, trace.json (the whole conversation), renders, outputs/
<stem>.py           the delivered CadQuery program
<stem>.stl          that program exported in the input mesh's frame
agent.log           console of the run
metrics.json        status, calls, tokens, time, how the run ended
```

Defaults are the paper's settings: temperature 0.2, 15k output tokens per
reply (a truncated reply is retried with up to 30k), a 250k-token context
budget, renders capped at 8 per command and 12 in the conversation (from the 6
most recent image-producing commands), outputs kept whole for 4 commands and
then reduced to their summaries. The command budget is `--max-calls` (default
30; the paper uses 30 for DeepCAD and BenchCAD and 60 for Fusion 360, MCB,
CADENA-Bench and BenchCAD-2). `--attempts N` retries a part that did not finish
and keeps the attempt with the smallest residual; `--skip-existing` resumes an
interrupted batch. `python agent_run.py --help` lists everything.

A run finishes when the residual (missing + extra volume against the input
mesh) is empty, or when the model concedes with the residual no longer falling.
The exit code is 0 when every part finished, 3 otherwise.

## Evaluation

```bash
# the tree agent_run.py wrote; GT = each part's own input mesh
python evaluate.py --predictions runs/ --out results/

# any method's flat <id>.stl predictions against a GT directory of <id>.stl
python evaluate.py --predictions preds/ --gt data/meshes/ --out results/

# optional per-group table (e.g. complexity tercile, part family)
python evaluate.py --predictions runs/ --groups groups.csv --out results/
```

It prints a table and writes `results/per_part.csv` and `results/results.json`.
Both meshes are placed in the ground truth's frame (GT bounding box centred,
longest side scaled to 1), then:

- **IoU** — volumetric IoU, boolean on closed bodies, voxel occupancy when
  either mesh is not watertight (`iou_method` says which);
- **GMS** — the Generalized Match Score of CADENA (8192 points, τ = 0.05,
  α_max = 25°), normalised to [0, 1];
- **CD** — symmetric Chamfer distance ×10³ on 8192 points (30k also reported);
- **IR** — invalid rate, the share of parts that produced no usable geometry.

IoU and GMS are averaged over valid parts; CD is the median over valid parts.
Predictions in the CADENA convention (`pred / 200 + 0.5` is the GT in the unit
cube) are scored with `--pred-frame unit200`.

## Citation

```bibtex
@misc{pavlov2026cadmageddon,
  title         = {CADMAGEDDON: High-Fidelity CAD Reconstruction with Local Inference},
  author        = {Pavlov, Oleg and Gavrilov, Nikita and Kabisov, Soslan and Savrasov, Gennadii
                   and Kuznetsov, Andrey and Konushin, Anton and Zhemchuzhnikov, Dmitrii},
  year          = {2026},
  eprint        = {2611.22563},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.22563}
}
```

## License

MIT — see [LICENSE](LICENSE). `metrics/gms.py` is vendored from
[CADENA](https://github.com/kulibinai/cadena) (MIT).
