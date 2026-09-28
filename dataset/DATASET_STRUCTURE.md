# Dataset Structure — M3-P2 (Speech Dataset + Mixture Generation)

## Folder layout

```
dataset/
├── metadata/
│   ├── speech_metadata.csv        # one row per clean speech clip (you provide)
│   ├── noise_metadata.csv         # one row per noise source clip (you provide)
│   ├── dataset_manifest.csv       # one row per generated mixture (script writes)
│   └── snr_manifest.csv           # split x noise_class x snr_db counts (script writes)
├── speech/                        # raw clean speech .wav files (16 kHz mono, you provide)
├── noise/                         # raw noise .wav files, by class (16 kHz mono, you provide)
└── mixtures/
    └── <split>/<noise_class>/<speaker>__<noise>__snr<value>.wav   # generated
```

`speech_metadata_TEMPLATE.csv` and `noise_metadata_TEMPLATE.csv` in
`dataset/metadata/` show the exact required columns — copy, fill with real
rows, rename (drop `_TEMPLATE`), and point `mixture_generator.py` at them.

## Speech diversity (metadata columns)

| Column | Required values | Purpose |
|---|---|---|
| `speaker_id` | any unique string per speaker | leakage-prevention grouping key |
| `gender` | `M` / `F` | ensures male/female coverage, per spec |
| `speaking_style` | `quiet` / `normal` / `loud` | vocal-effort diversity, per spec |
| `speaking_rate` | `slow` / `normal` / `fast` | speaking-rate diversity, per spec |
| `recording_condition` | free text (e.g. `quiet_room`, `vehicle_cabin`, `field`) | recording-condition diversity, per spec |

At minimum, aim for multiple speakers of each gender, and at least one clip
in each `speaking_style` and `speaking_rate` category, so the resulting
mixture set doesn't silently skew toward one speaking condition.

## Noise classes

Fixed to the six classes the spec lists: `engine`, `rotor`, `machinery`,
`wind`, `broadband`, `impulsive`. These deliberately mirror Member 2's
noise-classifier taxonomy where they overlap (engine/rotor/machinery) so the
two tracks can eventually share recordings — see `research/DATASET_RESEARCH.md`
for why no public dataset already provides these classes.

## Leakage prevention — dual axis (the "Critical" requirement)

The spec's requirement is: *same speaker/source must not appear in both
train and test.* This script enforces it on **both** sides of a mixture,
independently:

1. Every unique `speaker_id` is assigned to exactly one of train/val/test.
2. Every unique noise **source filename** is assigned to exactly one of
   train/val/test, completely independently of the speaker assignment.
3. A mixture is only ever built from a speech clip and a noise clip that
   landed in the *same* split — speech from a train-split speaker is never
   mixed with noise from a val- or test-split noise recording, or vice
   versa.

This is stronger than only splitting speakers (the more common approach in
speech-separation datasets like WHAM!/WHAMR!, which split speakers but reuse
the same overall noise pool across splits): if we only split speakers, a
model could still "memorize" a specific noise recording it saw during
training and get an inflated test score on that same recording. Splitting
independently on both axes closes that gap.

`mixture_generator.py` asserts zero overlap on both axes every time it runs
(`generate_mixtures()` raises immediately if either check fails) — this was
verified end-to-end on synthetic data before hand-off (see `README.md`).

## SNR range justification

The spec requires testing **-10, -5, 0, +5, +10, +15, +20 dB** and asks for
this range to be justified against SilentVox's intended environment, not
just copied from an existing benchmark. Two things point away from a
narrower range:

- **It's wider than existing speech datasets, deliberately.** WHAM!/WHAMR!
  use only -6 to +3 dB (urban ambient noise); the DNS Challenge's own
  synthesis and most DNS-derived training work use roughly -5 to +20 dB
  (see `research/DATASET_RESEARCH.md`). None of those were built for a
  weapon-adjacent, engine/rotor-heavy tactical environment.
- **The operational envelope is unusually wide.** Person 2's `Related Work`
  document documents PELTOR-/INVISIO-type passive+active attenuation in the
  28–43 dB SNR range under good conditions, and impulse peak levels of
  160–170 dBP at the source. In practice this means SilentVox has to stay
  useful across two very different regimes: (a) heavily degraded
  communication close to unattenuated engine/rotor noise or immediately
  after an impulsive event (justifying the -10 dB end), and (b) largely
  clean communication once passive/active attenuation is working well
  (justifying the +20 dB end). A narrower range tuned only to "typical"
  conditions would leave the enhancement stage unevaluated exactly where it
  matters most — right at the edge of usability.

The 5 dB step size (7 points from -10 to +20) balances evaluation
resolution against how many mixtures `mixture_generator.py` has to produce
per speech/noise pair — finer steps are a one-line change to
`SNR_LEVELS_DB` in `mixture_generator.py` if the team later wants denser
sampling around a specific operating point.

## Manifest schemas

### `dataset_manifest.csv` (one row per mixture)

| Column | Meaning |
|---|---|
| `mixture_id` | integer, unique per row |
| `speech_file`, `speaker_id`, `gender`, `speaking_style`, `speaking_rate`, `recording_condition` | copied from `speech_metadata.csv` |
| `noise_file`, `noise_class` | copied from `noise_metadata.csv` |
| `snr_db` | target SNR used to build this mixture |
| `split` | `train` / `val` / `test` |
| `output_path` | path to the generated mixture .wav, relative to `dataset/` |

### `snr_manifest.csv` (one row per split x noise_class x snr_db combination)

| Column | Meaning |
|---|---|
| `split`, `noise_class`, `snr_db` | grouping key |
| `count` | number of mixtures generated for that combination |

Use this to spot an imbalance (e.g. one noise class or SNR level
under-represented) before handing data off for training.

## Known limitations (stated honestly)

- The verification run in `README.md` used synthetic tone+noise audio, not
  real speech/noise recordings — the pipeline's *mechanics* (splitting,
  mixing, manifest generation) are verified; the *content* still depends on
  real recordings being added.
- `mixture_generator.py`'s WAV loader is stdlib+scipy only (no mp3/m4a) —
  same constraint as the P4 classifier pipeline; convert non-wav sources
  once during collection, not at mixture-generation time.
- Reverberation (vehicle cabin / earcup cavity acoustics) is not modelled
  here — that's a natural extension using the same synthetic-IR approach
  already implemented in Person 2's `augmentation.py`, not yet ported here.
