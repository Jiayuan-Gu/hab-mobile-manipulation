# Instructions

## Episode Generation

First, we need to enhance `habitat-lab/habitat/datasets/rearrange/rearrange_generator.py`:

Modify <https://github.com/facebookresearch/habitat-lab/blob/2ec4f6832422faebf20ca413b1ebf78547a4855d/habitat/datasets/rearrange/rearrange_generator.py#L1036>

```python
args, opts = parser.parse_known_args()
```

Add some codes to parse optional arguments <https://github.com/facebookresearch/habitat-lab/blob/2ec4f6832422faebf20ca413b1ebf78547a4855d/habitat/datasets/rearrange/rearrange_generator.py#L1050>

```python
if opts is not None:
    cfg.merge_from_list(opts)
```

Then, run `bash scripts/generate_episodes.sh` to generate episodes for Rearrangement tasks.

Finally, run `python scripts/merge_episodes.py` to merge generated episodes into a single file.
