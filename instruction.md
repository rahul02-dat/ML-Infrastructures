`/app/train.py` trains a small transformer under mixed-precision autocast, built on `TinyTransformer` from `/app/model.py` and batches produced by `/app/data.py`. Leave those two files as they are.

The script takes `--span_count`, the number of chunk-batches folded into a single optimizer update. With `--span_count 1` the run behaves the way you'd expect. At higher --span_count values, something breaks quietly: there are no exceptions or NaNs in the output, but the training dynamics and loss curve diverge noticeably from the baseline. Track down what in /app/train.py is responsible and correct it.

```
python3 /app/train.py --span_count N --out /app/logs/span_N.json
```

At steps 50, 100, 200, and 400 the script writes an entry into a JSON object (this schema is normative):

```json
{
  "span_count": 4,
  "loss_curve": {"50": 3.21, "100": 2.98, "200": 2.71, "400": 2.40},
  "grad_curve": {"50": 0.94, "100": 0.87, "200": 0.91, "400": 0.88},
  "scale_curve": {"50": 65536.0, "100": 65536.0, "200": 65536.0, "400": 65536.0},
  "commit_trace": [[1, t], [2, t], ...]
}
```

`loss_curve` records `step_loss` exactly as the script already accumulates it: the sum over the span's chunk-batches of each per-chunk loss (each already divided by `span_count`); do not rescale it. The loss-recording code is already correct; do not alter how it is calculated or reported. `grad_curve` is the norm of the gradient the optimizer actually applied at that step. `scale_curve` is the AMP scale factor in effect at that step. `commit_trace` is the wrapper's internal event trace: a list of `[step, t]` pairs appended to as the loop runs, where `t` is a running count the wrapper maintains internally at the moment of that event.

Once fixed, produce three runs — `--span_count 1`, `--span_count 4`, `--span_count 8` (leave `--steps`, `--chunk_batch`, and `--seed` at their defaults) — and write each one to:

- `/app/logs/span_1.json`
- `/app/logs/span_4.json`
- `/app/logs/span_8.json`