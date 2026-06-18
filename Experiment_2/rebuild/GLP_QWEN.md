# Qwen GLP On-Manifold Steering

This path is Qwen-only. It uses `Qwen/Qwen2.5-7B-Instruct`, hooks decoder block `20`, and projects edited activations at `hidden_states[21]` with hidden size `3584`.

Released GLP checkpoints are Llama-only and are rejected by the Qwen steering script. Train or provide a checkpoint whose `config.yaml` has `d_input: 3584` and `tracedict_config.layers: [20]`.

## 1. Build Qwen GLP Activations

Use a broad corpus for a useful manifold model:

```bash
python Experiment_2/rebuild/scripts/16_glp_qwen_cache.py \
  --source text_file \
  --input-texts /path/to/texts.txt \
  --all-tokens \
  --batch-size 8
```

For a tiny plumbing smoke test only:

```bash
python Experiment_2/rebuild/scripts/16_glp_qwen_cache.py \
  --source concept_pairs \
  --limit 64 \
  --chat-template \
  --batch-size 4
```

The default output is `Experiment_2/rebuild/cache/glp/qwen2_5_7b-layer20-static/`.

## 2. Train The Qwen GLP

From `Experiment_2/generative_latent_prior`:

```bash
python3 glp_train.py config=configs/train_qwen2_5_7b_static.yaml
```

The default checkpoint path is `Experiment_2/rebuild/workspace/glp_runs/glp-qwen2_5_7b-d6-static/final.safetensors`.

## 3. Run Qwen Steering

First make sure rebuild has Qwen stage-03 caches and directions:

```bash
python Experiment_2/rebuild/scripts/03_extract_for_battery.py --model qwen
python Experiment_2/rebuild/scripts/05_inject_matrix.py --model qwen
```

Then compare raw steering to GLP-projected steering:

```bash
python Experiment_2/rebuild/scripts/16_glp_steer_qwen.py \
  --mode both \
  --glp-weights Experiment_2/rebuild/workspace/glp_runs/glp-qwen2_5_7b-d6-static \
  --concepts refusal \
  --doses 0.55 \
  --n-anchors 5
```

Outputs are written to `Experiment_2/rebuild/results/gate4/` and examples to `Experiment_2/rebuild/workspace/glp_steering/`.

## CPU Checks

```bash
cd Experiment_2/rebuild
.venv/bin/python test_qwen_glp.py
```
