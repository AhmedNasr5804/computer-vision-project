# Artifacts Directory Description

This directory stores all generated outputs from the notebook pipeline, including:

- data manifests and split definitions,
- model binaries (.keras, .tflite, OpenVINO, SavedModel),
- experiment metrics JSON files,
- performance/deployment JSON files,
- diagnostic and paper-ready PNG figures.

The naming convention is stage-based:

- `01` data exploration
- `02` classical baseline
- `03` custom CNN baseline
- `04` transfer learning
- `05` improved experiments
- `06` model selection
- `07` fine-tuning
- `08` deployment

Prefixes:

- `eye*`: eye-state pipeline artifacts
- `lane*`: lane-direction pipeline artifacts

## File-by-file catalog

### Manifests and split definitions

- `00_manifest.json`: dataset paths, expected counts, and global reproducibility constants.
- `eye_split.json`: persistent train/val/test split definition for eye dataset.
- `lane_split.json`: persistent split definition for lane dataset.
- `eye_finetune_labels.json`: labels/metadata for phone eye fine-tuning samples.
- `lane_tusimple_index.pkl`: serialized index/cache for TuSimple lane data mapping and fast reuse.

### Global data-check figure

- `00_dataset_samples.png`: sanity panel showing representative samples from each source domain.

### Eye pipeline artifacts

#### Stage 01: data exploration

- `eye01_channel_hist.png`: eye dataset color/channel distribution diagnostics.
- `eye01_class_balance.png`: class-count balance for open vs. closed eyes.
- `eye01_mean_images.png`: per-class mean image visualizations.
- `eye01_samples_closed.png`: closed-eye sample montage.
- `eye01_samples_open.png`: open-eye sample montage.
- `eye01_size_hist.png`: image dimension/size distribution.

#### Stage 02: classical baseline

- `eye02_canny_demo.png`: Canny edge-processing illustration for eye pipeline.
- `eye02_hough_demo.png`: Hough feature demonstration (classical feature engineering step).
- `eye02_preprocess_demo.png`: preprocessing pipeline visualization.
- `eye02_cm_and_roc.png`: confusion matrix + ROC plot for classical eye baseline.
- `eye02_classical_results.json`: numeric metrics for classical eye model (accuracy, F1, and related stats).

#### Stage 03: custom CNN baseline

- `eye03_cnn_baseline.keras`: trained baseline Keras eye CNN.
- `eye03_cnn_baseline.tflite`: baseline eye CNN exported to TensorFlow Lite.
- `eye03_cnn_results.json`: quantitative results for baseline eye CNN.
- `eye03_train_curves.png`: training/validation learning curves.
- `eye03_overfit_gap.png`: overfitting/generalization-gap diagnostic.
- `eye03_cm_and_roc.png`: confusion matrix + ROC for baseline eye CNN.
- `eye03_misclassifications.png`: representative misclassified eye samples.

#### Stage 04: transfer learning

- `eye04_mobilenetv2_frozen.keras`: MobileNetV2 eye model with frozen backbone.
- `eye04_mobilenetv2_frozen.tflite`: TFLite export of frozen MobileNetV2 eye model.
- `eye04_mobilenetv2_finetune.keras`: MobileNetV2 eye model after fine-tuning.
- `eye04_mobilenetv2_finetune.tflite`: TFLite export of fine-tuned MobileNetV2 eye model.
- `eye04_mobilenetv3_small_frozen.keras`: MobileNetV3-Small eye model with frozen backbone.
- `eye04_mobilenetv3_small_frozen.tflite`: TFLite export of frozen MobileNetV3-Small eye model.
- `eye04_mobilenetv3_small_finetune.keras`: MobileNetV3-Small eye model after fine-tuning.
- `eye04_mobilenetv3_small_finetune.tflite`: TFLite export of fine-tuned MobileNetV3-Small eye model.
- `eye04_transfer_learning_results.json`: metrics table comparing all transfer-learning eye variants.
- `eye04_confusion_matrices.png`: confusion matrices for transfer-learning variants.
- `eye04_curves_overlay.png`: overlaid learning curves for transfer-learning variants.
- `eye04_params_vs_acc.png`: parameter-count versus accuracy trade-off chart.

#### Stage 05: improved experiments

- `eye05_experiments.json`: structured results from controlled eye experiments (augmentation/color/robustness/lightweight).
- `eye05_expA_aug.png`: experiment A visualization (augmentation study).
- `eye05_expB_color.png`: experiment B visualization (color/preprocessing variant).
- `eye05_expC_examples.png`: example outputs for experiment C.
- `eye05_expC_robustness.png`: robustness behavior visualization for experiment C.
- `eye05_expD_lightweight.png`: experiment D visualization (lightweight architecture focus).
- `eye05_lw_baseline.keras`: lightweight baseline eye model.
- `eye05_lw_baseline.tflite`: TFLite export of lightweight baseline eye model.
- `eye05_lw_tiny.keras`: tiny lightweight eye model variant.
- `eye05_lw_tiny.tflite`: TFLite export of tiny lightweight eye model.
- `eye05_lw_wide.keras`: wider lightweight eye model variant (selected winner family).
- `eye05_lw_wide.tflite`: TFLite export of wider lightweight eye model.

#### Stage 06: model selection

- `eye06_selection.json`: weighted-scoring and/or Pareto selection output for choosing eye winner.
- `eye06_pareto.png`: Pareto-front visualization for eye candidate models.
- `eye06_summary.png`: summary comparison panel for eye model-selection stage.
- `eye_winner.keras`: final selected eye model checkpoint.
- `eye_winner.tflite`: deployment-ready TFLite for selected eye model.

#### Stage 07: phone fine-tuning and detection visuals

- `eye07_face_detection.png`: phone-image face detection demonstration used in eye fine-tune workflow.
- `eye07_cropped_100x100.png`: normalized crop example showing eye fine-tune input format.
- `eye07_face_detection_user_only.png`: privacy-filtered or user-only face detection visualization.

#### Stage 08: deployment

- `eye08_deployment.json`: deployment benchmark and model-size summary for eye variants.
- `eye08_fp32.tflite`: float32 deployment build for eye model.
- `eye08_ptq_int8.tflite`: post-training quantized int8 eye deployment model.
- `eye08_s24_measured.png`: measured on-device performance on Samsung S24 Ultra.
- `eye08_variant_comparison.png`: comparison chart between eye deployment variants.

### Lane pipeline artifacts

#### Stage 01: data exploration

- `lane01_curvature_hist.png`: curvature-coefficient histogram used for direction bucket analysis.
- `lane01_direction_labels.png`: distribution/visualization of derived lane-direction labels.
- `lane01_direction_samples.png`: representative image samples per lane-direction class.
- `lane01_distributions.png`: additional lane data-distribution diagnostics.
- `lane01_finetune_samples.png`: samples from lane fine-tuning domain.
- `lane01_tusimple_overlays.png`: TuSimple lane overlays for annotation sanity checks.

#### Stage 02: classical baseline

- `lane02_classical_results.json`: classical lane baseline metrics.
- `lane02_confusion.png`: confusion matrix for lane classical model.
- `lane02_hough_demo.png`: probabilistic Hough transform demonstration for lane features.
- `lane02_preprocess_demo.png`: lane preprocessing pipeline demonstration.

#### Stage 03: custom CNN baseline

- `lane03_cnn_baseline.keras`: trained baseline Keras lane CNN.
- `lane03_cnn_baseline.tflite`: baseline lane CNN exported to TFLite.
- `lane03_cnn_results.json`: baseline lane CNN quantitative metrics.
- `lane03_confusion.png`: confusion matrix for lane CNN baseline.
- `lane03_misclassifications.png`: representative lane misclassifications.
- `lane03_train_curves.png`: learning curves for lane CNN baseline.

#### Stage 04: transfer learning

- `lane04_mobilenetv2_frozen.keras`: lane MobileNetV2 frozen-backbone variant.
- `lane04_mobilenetv2_frozen.tflite`: TFLite export of frozen MobileNetV2 lane model.
- `lane04_mobilenetv2_finetune.keras`: lane MobileNetV2 fine-tuned variant.
- `lane04_mobilenetv2_finetune.tflite`: TFLite export of fine-tuned MobileNetV2 lane model.
- `lane04_mobilenetv3_small_frozen.keras`: lane MobileNetV3-Small frozen-backbone variant.
- `lane04_mobilenetv3_small_frozen.tflite`: TFLite export of frozen MobileNetV3-Small lane model.
- `lane04_mobilenetv3_small_finetune.keras`: lane MobileNetV3-Small fine-tuned variant.
- `lane04_mobilenetv3_small_finetune.tflite`: TFLite export of fine-tuned MobileNetV3-Small lane model.
- `lane04_transfer_learning_results.json`: metrics table comparing lane transfer-learning variants.
- `lane04_confusion_matrices.png`: confusion matrices across transfer-learning lane models.
- `lane04_curves_overlay.png`: overlaid transfer-learning curves for lane models.
- `lane04_params_vs_acc.png`: parameters versus accuracy trade-off for lane transfer models.

#### Stage 05: improved experiments

- `lane05_experiments.json`: structured results from controlled lane experiments.
- `lane05_expA_aug.png`: experiment A visualization (augmentation effect).
- `lane05_expB_color.png`: experiment B visualization (color/preprocessing variant).
- `lane05_expC_examples.png`: experiment C sample outputs.
- `lane05_expC_robustness.png`: experiment C robustness behavior chart.
- `lane05_expD_lightweight.png`: experiment D lightweight-model comparison chart.
- `lane05_lw_baseline.keras`: lane lightweight baseline model.
- `lane05_lw_baseline.tflite`: TFLite export of lane lightweight baseline.
- `lane05_lw_tiny.keras`: lane tiny lightweight variant.
- `lane05_lw_tiny.tflite`: TFLite export of lane tiny lightweight model.
- `lane05_lw_wide.keras`: lane wide lightweight variant.
- `lane05_lw_wide.tflite`: TFLite export of lane wide lightweight model.

#### Stage 06: model selection

- `lane06_selection.json`: lane winner selection output based on scoring/efficiency criteria.
- `lane06_pareto.png`: Pareto-front chart for lane candidate models.
- `lane06_summary.png`: lane model-selection summary figure.
- `lane_winner.keras`: selected final lane model.
- `lane_winner.tflite`: selected lane deployment TFLite model.

#### Stage 07: Pi fine-tuning

- `lane07_finetune_results.json`: post-fine-tuning results on Pi domain.
- `lane07_confusion_pi.png`: confusion matrix for Pi-domain fine-tuned lane model.
- `lane07_sample_predictions.png`: sample predictions after lane fine-tuning.
- `lane_winner_finetuned.keras`: final lane model after Pi-domain fine-tuning.

#### Stage 08: deployment

- `lane08_deployment.json`: deployment metrics and package summary for lane variants.
- `lane08_fp32.tflite`: float32 lane deployment model.
- `lane08_ptq_int8.tflite`: int8 PTQ lane deployment model.
- `lane08_ov.xml`: OpenVINO IR network definition for lane deployment.
- `lane08_ov.bin`: OpenVINO IR weight binary paired with `lane08_ov.xml`.
- `lane08_rpi_simulated.png`: simulated Raspberry Pi latency/performance figure.
- `lane08_variant_comparison.png`: comparison plot across lane deployment variants.

### TensorFlow SavedModel export folder

- `_lane08_savedmodel/fingerprint.pb`: metadata fingerprint for SavedModel reproducibility.
- `_lane08_savedmodel/saved_model.pb`: SavedModel graph and signatures for lane deployment/export.
- `_lane08_savedmodel/assets/`: auxiliary assets folder (empty or optional runtime assets).
- `_lane08_savedmodel/variables/`: variable tensors used by the SavedModel.

## Usage guidance

- Use JSON files for exact numeric values in reports and reproducible comparisons.
- Use `.keras` as training/checkpoint artifacts.
- Use `.tflite` / OpenVINO outputs for edge-device deployment benchmarking.
- Use PNG files for diagnostics and paper figures.
