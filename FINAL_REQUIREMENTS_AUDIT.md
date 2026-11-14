# Final Requirements Audit

Audit date: 15 November 2026. Statuses reflect local artifacts, my report that GitHub upload and deployment fixes are complete, and the verification access available in this workspace. The new public URLs could not be fetched by the web checker during this update.

| Requirement | Evidence | Status | Remaining Action |
| --- | --- | --- | --- |
| Problem framing | README.md; final presentation slides 2–3 | PASS | None |
| EDA and evidence-based insights | Notebook EDA cells and rendered plots; README; final presentation slide 5 | PASS | None |
| Notebook execution | All 7 notebook code cells executed sequentially from the repository root in one shared Python namespace; execution completed without exceptions and reproduced the final metrics, reports and saved model. Jupyter UI `Run All` was not separately tested. | PASS | Optional: open the notebook in Jupyter and use Restart Kernel & Run All. |
| Data understanding, source, class balance, missing data and duplicates | Notebook; reports/test_metrics.json; README; slide 4 | PASS | None |
| Preprocessing and leakage prevention | src/train.py; notebook; joblib pipeline; README; slide 6 | PASS | None |
| Modeling and baseline | reports/model_comparison.csv; notebook; slide 7 | PASS | None |
| Hyperparameter tuning | Notebook RandomizedSearchCV cells; reports/model_comparison.csv best_params; slide 8 | PASS | None |
| Cross-validation | Five-fold StratifiedKFold on training data; CSV CV F1 values; slide 8 | PASS | None |
| Held-out evaluation and model comparison | reports/model_comparison.csv and reports/test_metrics.json; slide 9 | PASS | None |
| Error analysis | reports/test_metrics.json confusion matrix; notebook error examples; slide 10 | PASS | None |
| Business recommendations and limitations | README; notebook; final presentation slide 11 | PASS | None |
| Saved model and dependency compatibility | Joblib estimator state embeds scikit-learn 1.7.2. The model loads in local Python 3.12.0/scikit-learn 1.7.2; preprocessing and probability scoring pass. `requirements.txt` now pins the direct runtime/training dependencies. | PASS | Install these pins in the Cloud environment. |
| Local Streamlit application and model schema | Updated `app.py`; matching 16-feature input schema; AppTest submits Predict without exceptions and displays class plus probability. Version and full exception details are shown for diagnosis. | PASS | None for local version. |
| Public deployment | I fixed the app errors and supplied the new live URL. The URL could not be independently opened or submitted to from this workspace; Cloud runtime caption was not observed. | NOT TESTED | Confirm the app loads, submit Predict, and verify the visible runtime says Python 3.12.x/scikit-learn 1.7.2. |
| README completeness and image paths | Local README includes required sections, the new exact app URL, GitHub link, CSV metrics, confusion matrix, author GitHub handle and relative screenshots; both image files exist. | PASS | Upload this updated README to GitHub. Add a full name/LinkedIn profile only if desired. |
| PowerPoint content, slide count and visual review | purchase_intent_presentation_FINAL.pptx; 12 rendered slides inspected; editable charts on slides 4, 5 and 8; native tables on slides 9–10; package/layout checks report zero findings. | PASS | Re-render after any edits. PowerPoint itself was not launched for native application testing. |
| GitHub repository publication | I uploaded the project files to [Safaa-cloud/ecommerce-purchase-intent-prediction](https://github.com/Safaa-cloud/ecommerce-purchase-intent-prediction), except README. The repository could not be independently fetched from this workspace. | PARTIAL | Upload the updated README and updated audit/LinkedIn files, then confirm the repository contents. |
| Actual app screenshots | screenshots/app_interface.png and screenshots/prediction_result.png exist, resolve from README, and were visually inspected. | PASS | No missing screenshot. Refresh the prediction screenshot after the hosted prediction works. |
| LinkedIn post draft | LINKEDIN_POST_DRAFT.md contains the professional opening, problem, dataset, tools, actual measured results, lesson, new app link and GitHub repository link. | PASS | Upload the updated draft if keeping it in sync on GitHub; publish manually only if desired. |
| CV bullet | CV_PROJECT_BULLET.md uses verified dataset scale, model results and technologies. | PASS | Add the bullet to the CV if desired. |
| Author information | The README links my GitHub handle to my profile. | PARTIAL | Add my full name and LinkedIn profile if I choose to display them. |
| `.gitignore` and upload readiness | `.gitignore` excludes environments, caches, secrets, temp builds and the raw CSV; model artifact is retained for the app. | PASS | Keep the raw data file out of the manual GitHub upload. |

## Verified result values

The completed training run and reports agree. The selected model is Random Forest, selected by the highest training-only CV F1 of 0.417635. At threshold 0.50 its held-out test scores are precision 0.300826, recall 0.476440, F1 0.368794, ROC-AUC 0.754845 and PR-AUC 0.342545. The confusion matrix is `[[1661, 423], [200, 182]]`.

The comparison CSV also shows Extra Trees had the highest test F1 (0.401104) on this particular split. Random Forest remains the selected model because selection used cross-validation F1 on the training data, not test-set optimization.

## Remaining completion items

1. Upload the finished local `README.md` and the updated `FINAL_REQUIREMENTS_AUDIT.md` and `LINKEDIN_POST_DRAFT.md` to GitHub so the repository matches this workspace. I uploaded the other project files.
2. Open the new Streamlit URL, submit the default prediction, and confirm the app displays a result and a Python 3.12.x / scikit-learn 1.7.2 runtime caption. This environment could not access the app URL, so deployment remains NOT TESTED here.
3. Add a preferred full name and LinkedIn profile in README if desired.

I fixed the live app and uploaded the project files; README is the one file still missing from GitHub. Independent remote verification remains outstanding.
## Compatibility verification note

Every pinned direct dependency version in `requirements.txt` matches the active local environment. A clean virtual-environment installation was attempted, but package-index DNS/network access is unavailable in this workspace, so a fresh install from the file could not be verified here. The app and model were tested in the matching installed environment instead.
