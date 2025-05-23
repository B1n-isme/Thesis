- input_size: number of read past points

- step_size: slide step, if step_size >= input_size -> no overlap

- windows_batch_size: process input_size together in batch then average out mistake
    -> too small: noisy
    -> too large: takes longer learning

- max_steps: num of train iterations (num_epoch)
- val_check_steps: freq in making validate -> smaller is better since spot quick early stopping but can be computation overhead
    -> these 2 often go with early_stop_patience_steps
