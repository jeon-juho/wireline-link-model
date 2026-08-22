# Channel data

The S-parameter files this model runs on are **not stored in this repository**.
They are contributed materials of the IEEE P802.3ck task force, so they are
downloaded from the source rather than redistributed here.

## Download

**Archive:** [`palkert_3ck_02_0120.zip`](https://www.ieee802.org/3/ck/public/tools/backplane/palkert_3ck_02_0120.zip) — 10 MB

Contributed by Tom Palkert (Molex), January 2020. The accompanying presentation,
*BP OD Channel Analysis*, is
[`palkert_3ck_01a_0120.pdf`](https://www.ieee802.org/3/ck/public/20_01/palkert_3ck_01a_0120.pdf),
and the full index of task-force channel contributions is at
[ieee802.org/3/ck/public/tools](https://www.ieee802.org/3/ck/public/tools/).

## Install

Extract the archive into this directory, so that the paths look like:

```
channel/
└── palkert_3ck_02_0120/
    ├── Asymmetric_channel_16inch_16inch/
    │   ├── THRU_VL5_OD-BP-Channel_16inch_16inch.s4p     <-- the reference channel
    │   ├── NEXT_VL5_A1_OD-BP-Channel_16inch_16inch.s4p
    │   ├── FEXT_VL5_A4_OD-BP-Channel_16inch_16inch.s4p
    │   └── ...
    ├── Asymmetric_channel_8inch_24inch/
    ├── Asymmetric_channel_5inch_27inch/
    └── Asymmetric_channel_4inch_28inch/
```

Then check it resolved:

```bash
python scripts/plot_channel.py
```

`src/config.py` looks for the reference THRU file in a few plausible layouts, so
an extra or missing nesting level is tolerated. If your extraction ended up
somewhere else entirely, point at it directly instead:

```bash
python scripts/plot_channel.py --channel path/to/THRU_VL5_OD-BP-Channel_16inch_16inch.s4p
```

## What the archive contains

Four backplane cases, each a 16-inch-equivalent link split differently between
the two daughter cards (16+16, 8+24, 5+27, 4+28 inches). Every case has a THRU
channel plus three NEXT and four FEXT aggressors.

The results in the top-level README all use
**`Asymmetric_channel_16inch_16inch/THRU_VL5_OD-BP-Channel_16inch_16inch.s4p`**:
0–50 GHz, 2501 points on a uniform 20 MHz grid with DC included, 50 Ω
single-ended, 4-port. Its differential insertion loss is **13.19 dB at the 8 GHz
Nyquist frequency** and its propagation delay 4.91 ns.

Port ordering is **through-paired** — ports (1,2) are the two ends of one line
and (3,4) the two ends of the other, so the differential pairs are {1,3} at the
near end and {2,4} at the far end. `src/channel.py` detects this from the data
rather than assuming it, because assuming wrongly produces a plausible-looking
result computed from the near-end coupling instead of the through path.

The NEXT and FEXT files are unused so far: the model does not yet include
crosstalk, which `docs/spec.md` §7 flags as the assumption that makes every
margin here optimistic. They are the raw material for fixing that.
