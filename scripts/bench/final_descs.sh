# Final descriptors (selected on val; docs/s1_findings.md, docs/s2_findings.md). Sourced by the S3 scripts.
A_GEO=outputs/train/s1_geo_lr1e-4/step_008000.pt      # arm A (Sonata), geometry-only training
A_MM=outputs/train/s2_mm_lam0_lr1e-4/step_010000.pt   # arm A, multimodal (lambda 0)
B_GEO=outputs/train/s1_ppf/step_008000.pt             # arm B (PPF), geometry-only training
B_MM=outputs/train/s2_ppf_lam0/step_008000.pt         # arm B, multimodal (lambda 0)
LEARNED=("learned:$A_GEO" "learned:$A_MM" "learned:$A_MM:geo" "learned:$B_GEO" "learned:$B_MM" "learned:$B_MM:geo")
BASELINES=(fpfh_r20 sonata sonata_pca32 fpfh_r20+dino)
