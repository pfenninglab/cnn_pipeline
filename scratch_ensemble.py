

import models

model_paths = [
	"/home/csestili/repos/mouse_sst_clean/wandb/run-20220605_234548-3qqe7wvu/files/model-best.h5",
	"/home/csestili/repos/mouse_sst_clean/wandb/run-20220603_080815-v0r62029/files/model-best.h5",
	"/home/csestili/repos/mouse_sst_clean/wandb/run-20220605_101417-6jj0tsap/files/model-best.h5"
]

ensemble, metrics = models.get_bagged_ensemble(model_paths)

ensemble.summary()
