#!/usr/bin/env bash
# =============================================================================
# analysis/make_paper_figures.sh
# =============================================================================
# Every figure and table of the paper, body and appendix.
#
#     RESULTS_ROOT=/home/a.dorizza/OUTPUTS_MEDICAL \
#         bash analysis/make_paper_figures.sh [OUT_DIR] [CACHE_DIR]
#
# Environment:
#   RESULTS_ROOT  OUTPUTS_MEDICAL root (required)
#   REFRESH=1     re-read the results tree instead of the cache (needed after
#                 any rerun of a stage)
#   QUALITATIVE=1 also draw the qualitative maps (needs a GPU and checkpoints);
#                 QUAL_INDEX=<n> picks the test image (default 0)
#   DATASETS=1    also emit Table 1 and the dataset appendix (needs the images)
#
# Final PDFs are copied to OUT_DIR/paper_figures/{main,appendix} and the
# appendix tables to OUT_DIR/paper_tables/appendix, with the names and layout
# of the Overleaf project. A script whose verification fails stops the driver.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

: "${RESULTS_ROOT:?set RESULTS_ROOT}"
export RESULTS_ROOT
OUT="${1:-paper_results}"
CACHE="${2:-.cache_analysis}"
FIGS="${OUT}/paper_figures"
mkdir -p "${FIGS}"

run() { echo; echo ">>> $*"; python -m "$@"; }
C="--results_root ${RESULTS_ROOT} --cache ${CACHE}"

# Read the results tree once; every later call reuses the cache.
FIRST_REFRESH=""
[ "${REFRESH:-0}" = "1" ] && FIRST_REFRESH="--refresh"

# -- 0. Clean metrics (Tab. 3) and head spectral norm (Fig. 2) ----------------
run analysis.finetuning.create_csv
OUT_DIR="${OUT}" run analysis.finetuning.generate_tables
OUT_DIR="${OUT}" run analysis.finetuning.generate_figures

# -- Lambda ablation tables (Appendix), from the ablation runs ------------------
echo; echo ">>> ablation/select_lambda.py"
python3 ablation/select_lambda.py "${RESULTS_ROOT}/ablation" --outdir "${OUT}/ablation"

# -- 1. Leakage with depth ----------------------------------------------------
run analysis.1_depth_leakage ${C} ${FIRST_REFRESH} --out "${OUT}/1_depth" \
    --task both --layout stacked

# -- 2. Inside / outside drift as predictors ----------------------------------
# Body: segmentation row on the 'none' stratum. 'overall' kept for comparison.
for stratum in none overall; do
    run analysis.2_drift_predicts_damage ${C} --out "${OUT}/2_drift" \
        --task both --ratio all --seg_stratum "${stratum}"
done

# -- 3. Severity and corrupted area -------------------------------------------
for task in both classification segmentation; do
    run analysis.3_in_out_dose_response ${C} --out "${OUT}/3_in_out" --task "${task}"
done

# -- 4. Mask geometry ---------------------------------------------------------
for task in both classification segmentation; do
    run analysis.4_geometry_beats_area ${C} --out "${OUT}/4_geometry" --task "${task}"
done

# -- 5. Clean and corrupted performance ---------------------------------------
D="${OUT}/5_adaptation"
run analysis.5_adaptation_not_robustness ${C} --out "${D}" --task classification \
    --no_legend --row_label classification
run analysis.5_adaptation_not_robustness ${C} --out "${D}" --task segmentation \
    --row_label segmentation
run analysis.stack_tasks "${D}/adaptation_not_robustness_classification.pdf" \
    "${D}/adaptation_not_robustness_segmentation.pdf" \
    -o "${D}/adaptation_not_robustness_stacked.pdf"

# -- 6. Margin certificate ----------------------------------------------------
D="${OUT}/6_certificate"
for pert in gaussian Contrast; do
    run analysis.6_certificate_needs_sigma ${C} --out "${D}" --task classification \
        --perturbation "${pert}" --no_legend --row_label classification
    run analysis.6_certificate_needs_sigma ${C} --out "${D}" --task segmentation \
        --perturbation "${pert}" --row_label segmentation
    run analysis.stack_tasks "${D}/certificate_needs_sigma_classification_${pert}.pdf" \
        "${D}/certificate_needs_sigma_segmentation_${pert}.pdf" \
        -o "${D}/certificate_needs_sigma_${pert}_stacked.pdf"
done

# -- Appendix E: MSE instead of the angular distortion -------------------------
D="${OUT}/appendix_mse"
run analysis.1_depth_leakage ${C} --out "${D}" --task both --layout stacked --metric mse
# The checks of script 2 are stated for the angular drift: under MSE they are
# reported but do not stop the driver.
run analysis.2_drift_predicts_damage ${C} --out "${D}" --task both --ratio all \
    --seg_stratum none --metric mse || echo "  [note] MSE: see the checks above"
run analysis.3_in_out_dose_response ${C} --out "${D}" --task both --metric mse

# -- Appendix F and G ---------------------------------------------------------
run analysis.appendix.per_dataset ${C} --out "${OUT}/appendix_per_dataset"
run analysis.appendix.full_grid ${C} --out "${OUT}/appendix_grid"
# Corruption audit (Appendix C): real test images with DATASETS=1, else synthetic.
AUDIT_IMAGES=""
[ "${DATASETS:-0}" = "1" ] && AUDIT_IMAGES="--dataset kvasir_seg"
run analysis.appendix.corruption_audit ${C} --out "${OUT}/appendix_corruption" ${AUDIT_IMAGES}
# Needs the outputs of scripts 2, 4 and 6 and of per_dataset, all written above.
run analysis.appendix.claims_per_dataset --out "${OUT}"

# -- Optional: qualitative maps (Section 5.5 and appendix E) ---------------------
# QUAL_INDEX = test-set position of the image (chosen with select_qualitative.py).
if [ "${QUALITATIVE:-0}" = "1" ]; then
    QI="${QUAL_INDEX:-0}"
    run analysis.qualitative_maps --results_root "${RESULTS_ROOT}" \
        --out "${OUT}/qualitative" --dataset kvasir_v2 --index "${QI}" \
        --archs vit resnet --ptypes "Gaussian Noise" --style overlay --name qualitative_main
    run analysis.qualitative_maps --results_root "${RESULTS_ROOT}" \
        --out "${OUT}/qualitative" --dataset kvasir_v2 --index "${QI}" \
        --archs vit resnet dinov2 --ptypes "Gaussian Noise" "Contrast" --style overlay \
        --name qualitative_all
fi
if [ "${DATASETS:-0}" = "1" ]; then
    run analysis.appendix.datasets --out "${OUT}/appendix_datasets"
fi

# -- Copy the PDFs the paper includes, with the Overleaf layout --------------
# OUT_DIR/paper_figures/{main,appendix} mirrors figs/{main,appendix} in the
# Overleaf project: upload the two folders as they are.
mkdir -p "${FIGS}/main" "${FIGS}/appendix"
copy() {
    if [ -f "$1" ]; then cp "$1" "${FIGS}/$2"; echo "  ${FIGS}/$2"
    else echo "  [missing] $1"; fi
}
echo; echo "paper figures:"
copy "${OUT}/finetuning/head_spectral_norm.pdf"                          main/0_head_spectral_norm.pdf
copy "${OUT}/1_depth/depth_leakage_both_stacked.pdf"                     main/1_depth_leakage.pdf
copy "${OUT}/2_drift/drift_predicts_damage_r0.3-0.6-0.9_seg-none.pdf"    main/2_drift_predicts_damage.pdf
copy "${OUT}/3_in_out/in_out_dose_response_both_cos.pdf"                 main/3_dose_response.pdf
copy "${OUT}/4_geometry/geometry_beats_area_both.pdf"                    main/4_geometry.pdf
copy "${OUT}/5_adaptation/adaptation_not_robustness_stacked.pdf"         main/5_adaptation.pdf
copy "${OUT}/6_certificate/certificate_needs_sigma_gaussian_stacked.pdf" main/6_certificate_gaussian.pdf
copy "${OUT}/3_in_out/in_out_dose_response_classification_cos.pdf"       appendix/3_dose_response_classification.pdf
copy "${OUT}/3_in_out/in_out_dose_response_segmentation_cos.pdf"         appendix/3_dose_response_segmentation.pdf
copy "${OUT}/4_geometry/geometry_beats_area_classification.pdf"          appendix/4_geometry_classification.pdf
copy "${OUT}/4_geometry/geometry_beats_area_segmentation.pdf"            appendix/4_geometry_segmentation.pdf
copy "${OUT}/6_certificate/certificate_needs_sigma_Contrast_stacked.pdf" appendix/6_certificate_contrast.pdf
copy "${OUT}/appendix_mse/depth_leakage_both_stacked_mse.pdf"            appendix/E_leakage_mse.pdf
copy "${OUT}/appendix_mse/drift_predicts_damage_r0.3-0.6-0.9_mse_seg-none.pdf" appendix/E_predicts_mse.pdf
copy "${OUT}/appendix_mse/in_out_dose_response_both_mse.pdf"             appendix/E_dose_response_mse.pdf
copy "${OUT}/appendix_per_dataset/per_dataset_retained.pdf"              appendix/F_per_dataset_retained.pdf
copy "${OUT}/qualitative/qualitative_main_overlay.pdf"                   main/qualitative.pdf
copy "${OUT}/qualitative/qualitative_all_overlay.pdf"                    appendix/qualitative_all.pdf

# -- Copy the tables the paper \inputs, with the Overleaf layout --------------
# OUT_DIR/paper_tables/{main,appendix} mirrors tables/{main,appendix}.
TABS="${OUT}/paper_tables"
mkdir -p "${TABS}/main" "${TABS}/appendix"
tab() {
    if [ -f "${OUT}/$1" ]; then cp "${OUT}/$1" "${TABS}/$2"; echo "  ${TABS}/$2"
    else echo "  [missing] ${OUT}/$1"; fi
}
echo; echo "paper tables:"
tab finetuning/clean_metrics.tex                     main/clean_metrics.tex
tab ablation/ablation_lambda_classification.tex      appendix/ablation_lambda_classification.tex
tab ablation/ablation_lambda_segmentation.tex        appendix/ablation_lambda_segmentation.tex
tab ablation/ablation_lambda_tol_sensitivity.tex     appendix/ablation_lambda_tol_sensitivity.tex
tab appendix_per_dataset/claims_per_dataset.tex      appendix/claims_per_dataset.tex
tab appendix_grid/full_grid_classification.tex       appendix/full_grid_classification.tex
tab appendix_grid/full_grid_segmentation.tex         appendix/full_grid_segmentation.tex
tab appendix_corruption/corruption_audit.tex         appendix/corruption_audit.tex
if [ "${DATASETS:-0}" = "1" ]; then
    tab appendix_datasets/datasets.tex               main/datasets.tex
    for t in classes_kvasir_v2 classes_hyperkvasir classes_kvasir_capsule \
             classes_dropped aircare_sites; do
        tab "appendix_datasets/${t}.tex"             "appendix/${t}.tex"
    done
fi
