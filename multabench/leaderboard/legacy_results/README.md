# Legacy results

The results reported in the paper, kept for reference while every model is rerun under the leaderboard protocol.
New results go to `multabench/leaderboard/models/`, one CSV per model; these files are not updated anymore.

They differ from the leaderboard runs in how they were produced:

- 5 folds instead of 3 folds × 2 repeats, with other splits.
- `test_score` is ROC AUC for classification and R² for regression, not the TabArena metrics.
- 40 datasets: the 20 text and 20 image datasets of the paper (`text/`, `images/`), with `more_baselines/` for the
  other models and `text_large/`, `images_large/` for the e5-large and DINOv3-large encoders.
- `multimodal_state` is `all` for frozen encoders and `ft` for encoders fine-tuned on the task.

On the 40 datasets both cover, the frozen (`all`) results agree with the leaderboard runs of the same 11 models:
the Elo order has a Spearman correlation of 0.998, the order of the models on each dataset a mean Kendall τ of 0.83,
and the ROC AUC on binary datasets is within 0.012 per model on average.
