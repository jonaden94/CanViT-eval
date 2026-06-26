# Single source of truth for exp21 evaluation configs: "RUN|OVERRIDE_SCALE".
# Sourced by in1k_clf.sbatch and reconstruction.sbatch (array index -> entry).
#
# 32 configs = 23 base (one per run) + 9 special (3x3 scale sweep).
#
# OVERRIDE_SCALE:
#   none  -> pass the policy's own coarse-to-fine scales (full 1 -> 0.5 -> 0.25)
#   <flt> -> pin every glimpse to that scale, keeping the policy's centers
#            (foveated/square: fix_size = scale*H; uniform: pre-crop zoom)
#
# Policy is coarse_to_fine, n_timesteps=21 (first 3 quadtree levels: 1+4+16),
# canvas_grid=32 (the single grid the standardizers were trained at).
#
# Teacher + IN1k probe are auto-selected per run from the checkpoint's recorded
# teacher_name (ViT-B vs ViT-L), so the 2 ViT-L-distilled runs (cond4-fovi-L,
# uniform16-L) evaluate against the DINOv3 ViT-L teacher + ViT-L probe.
EVAL_CONFIGS=(
  # ===== BASE: uniform patcher -> policy (coarse-to-fine) scales =====
  "exp21-uniform8|none"
  "exp21-uniform16-grid4|none"
  "exp21-uniform16-repro|none"
  "exp21-uniform-p6-grid5|none"
  "exp21-uniform-p6-grid6|none"
  "exp21-uniform-overlap-p16-grid7-stride8|none"
  "exp21-uniform16-L|none"                              # ViT-L teacher (auto)

  # ===== BASE: foveated patcher -> full-image scale 1.0 =====
  "exp21-cond4-fovi-film-pos-sigma4-repro|1.0"
  "exp21-cond4-fovi-film-pos-sinusoidal|1.0"
  "exp21-cond4-fovi-fov140-res81|1.0"
  "exp21-cond4-fovi-fov140-res81-force|1.0"
  "exp21-cond4-fovi-L|1.0"                              # ViT-L teacher (auto)

  # ===== BASE: square patcher -> full-image scale 1.0 (scale1p41 -> 1.41) =====
  "exp21-cond4-reg-film-pos-sigma4|1.0"
  "exp21-cond4-reg-modulate-trunk-crossattn-fourier|1.0"
  "exp21-cond4-reg-modulate-trunk-fourier|1.0"
  "exp21-cond4-reg-modulate-trunk-sinusoidal|1.0"
  "exp21-cond4-reg-prune30|1.0"
  "exp21-cond4-reg-repro|1.0"
  "exp21-cond4-reg-scale-uniform-glimpse|1.0"
  "exp21-cond4-reg-scale-uniform-rollout|1.0"
  "exp21-cond4-reg-scale1p41|1.41"
  "exp21-strided-gf2|1.0"
  "exp21-strided-gf2-keepcorners|1.0"

  # ===== SPECIAL: 3x3 scale sweep {glimpse, rollout, repro} x {none, 0.6, 0.8} =====
  # per-glimpse model (trained on per-glimpse scales in [0.05,1.41])
  "exp21-cond4-reg-scale-uniform-glimpse|none"
  "exp21-cond4-reg-scale-uniform-glimpse|0.6"
  "exp21-cond4-reg-scale-uniform-glimpse|0.8"
  # per-rollout model (trained on one constant scale per rollout)
  "exp21-cond4-reg-scale-uniform-rollout|none"
  "exp21-cond4-reg-scale-uniform-rollout|0.6"
  "exp21-cond4-reg-scale-uniform-rollout|0.8"
  # scale-1-only control (same architecture, trained only at scale 1 -> OOD here)
  "exp21-cond4-reg-repro|none"
  "exp21-cond4-reg-repro|0.6"
  "exp21-cond4-reg-repro|0.8"
)

# Parse EVAL_CONFIGS[$1] -> sets globals RUN, OVR_CLI, TAG.
#   OVR_CLI: value passed to --episode.override-scale ("None" or the float)
#   TAG:     output-filename scale tag ("policy", "s1p0", "s0p6", "s1p41", ...)
parse_eval_config() {
  local entry="${EVAL_CONFIGS[$1]}"
  RUN="${entry%%|*}"
  local ovr="${entry##*|}"
  if [ "$ovr" = "none" ]; then
    OVR_CLI="None"; TAG="policy"
  else
    OVR_CLI="$ovr"; TAG="s${ovr//./p}"
  fi
}
