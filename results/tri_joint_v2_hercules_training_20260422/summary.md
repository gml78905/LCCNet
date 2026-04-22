# TriJointV2 Hercules Training Summary

## Experiment

- Name: `tri_joint_v2_hercules`
- Date: `2026-04-22`
- Dataset: `Hercules`
- Train scenes: `["SC_1", "SC_3", "island_1"]`
- Val scenes: `["library_1"]`
- Model: `TriJointV2`
- Task: tri-modal pairwise delta extrinsic calibration

## Run Setup

- Epochs: `120`
- Train iterations per epoch: `266`
- Val iterations per epoch: `84`
- Sequence mode: enabled
- Sequence length: `4`
- Input resolution: `288 x 512`
- Batch size: `36`
- Number of workers: `8`
- AMP: enabled (`fp16`)
- Point cloud cache: enabled

## Final Runtime

- Full training time: `12.19 hr`
- Typical epoch time near convergence: `351-358 sec`
- Final Sacred result: `0.38745225773559316`

## Best Checkpoint

- Best validation epoch: `112`
- Best validation loss: `0.387`
- Best checkpoint:
  - `/workspace/data/checkpoints/LCCNet/hercules/tri_joint_v2_hercules/models/checkpoint_tri_best.tar`

## Best Validation Metrics

### Input error

- Translation cm:
  - `CL = 47.991`
  - `CR = 49.225`
  - `LR = 68.199`
- Rotation deg:
  - `CL = 4.791`
  - `CR = 4.805`
  - `LR = 6.613`

### Corrected error at best epoch (`112`)

- Translation cm:
  - `CL = 18.106`
  - `CR = 40.785`
  - `LR = 44.604`
- Rotation deg:
  - `CL = 0.916`
  - `CR = 3.700`
  - `LR = 3.932`

## Late-Epoch Validation Trend

- `epoch 111`: `val loss = 0.435`
- `epoch 112`: `val loss = 0.387` ← best
- `epoch 113`: `val loss = 0.433`
- `epoch 114`: `val loss = 0.451`
- `epoch 115`: `val loss = 0.448`
- `epoch 116`: `val loss = 0.499`
- `epoch 117`: `val loss = 0.542`
- `epoch 118`: `val loss = 2.442`
- `epoch 119`: `val loss = 2.312`
- `epoch 120`: `val loss = 0.465`

## Training Stability

- Training loss after convergence stayed near `0.238-0.243`
- Validation performance peaked around `epoch 112`
- A sharp validation degradation appeared at `epoch 118-119`
- Validation partially recovered at `epoch 120`

## Pairwise Interpretation

### CL

- Strongest pair in the run
- Large improvement over input perturbation
- Best epoch reached:
  - `18.106 cm`
  - `0.916 deg`

### CR

- Improved over the input, but much weaker than CL
- Best epoch reached:
  - `40.785 cm`
  - `3.700 deg`
- Radar-related generalization remains limited

### LR

- Improved over input, but remains the weakest pair together with CR
- Best epoch reached:
  - `44.604 cm`
  - `3.932 deg`
- The radar-involved relation remains difficult

## Interpretation

This run is meaningful and should be kept.

Key takeaways:

- `TriJointV2` is trainable end-to-end on Hercules and converges stably on training loss.
- `CL` performance is clearly improved and is the strongest pair.
- `CR` and `LR` improve, but remain significantly weaker than `CL`.
- The run shows that the new architecture is viable, but radar-related pair generalization is still the main bottleneck.
- Late-stage validation instability suggests that the best checkpoint should be selected earlier rather than relying on the final epoch.

## Conclusion

- Status: `keep`
- Use `checkpoint_tri_best.tar` from `epoch 112` for comparison and follow-up evaluation.
- Do not use the final checkpoint as the representative result.

## Recommended Next Actions

1. Compare this run against the previous direct-regression or V1 baseline using the same pairwise metrics.
2. Analyze why `CR` and `LR` remain weak despite the tri-joint reliability-aware structure.
3. Add stronger monitoring for validation instability after the best epoch.
4. Consider earlier stopping or best-checkpoint-first evaluation in future long runs.
