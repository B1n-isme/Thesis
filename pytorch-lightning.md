
Ran tool
Here are the main updated and new functions in the latest stable version of PyTorch Lightning (2.5.x), focusing on changes since 2.0 and especially those relevant for users upgrading or seeking new features:

---

### **Key Updated/New Functions and Features**

#### **1. LightningModule API**
- **configure_model**: New hook for advanced model sharding and parallelism (e.g., FSDP, tensor parallelism).
- **on_train_epoch_end / on_validation_epoch_end**: Use these instead of the removed `*_epoch_end` hooks for epoch-level operations.
- **manual optimization**: Multiple optimizers now require manual optimization mode (`self.automatic_optimization = False`).
- **CombinedLoader**: For multi-dataloader support, use `CombinedLoader` with new modes (`min_size`, `max_size_cycle`, `max_size`, `sequential`).

#### **2. Trainer API**
- **barebones mode**: `Trainer(barebones=True)` disables all non-essential features for speed benchmarking.
- **devices/accelerator/strategy**: Now default to `"auto"`, simplifying hardware selection.
- **resume from checkpoint**: Use `trainer.fit(model, ckpt_path=...)` instead of `resume_from_checkpoint` argument.
- **use_distributed_sampler**: Replaces `replace_sampler_ddp`.

#### **3. Precision and Distributed Training**
- **Precision settings**: Use `"16-mixed"`, `"bf16-mixed"`, `"32-true"`, `"64-true"` for clarity.
- **Native FSDP**: Only native PyTorch FSDP is supported; FairScale is removed.
- **No DataParallel**: DP strategy is removed; use DDP or FSDP.

#### **4. Logging and Callbacks**
- **self.log_dict**: Log multiple metrics at once.
- **on_exception checkpoint**: New callback to save checkpoints on exceptions.
- **No automatic loss in progress bar**: Explicitly log loss with `self.log("loss", loss, prog_bar=True)`.

#### **5. Removed/Changed**
- **Removed**: `training_epoch_end`, `validation_epoch_end`, `test_epoch_end`, `training_step_end`, `validation_step_end`, `test_step_end`.
- **Removed**: `Trainer.tune()`, `auto_lr_find`, `auto_scale_batch_size` (use `Tuner`).
- **Removed**: `track_grad_norm` (use `on_before_optimizer_step` + `grad_norm` utility).
- **Removed**: `Trainer.add_argparse_args`, `from_argparse_args` (use LightningCLI or manual argparse).

---

### **Notable Additions in 2.4/2.5**

- **ModelParallelStrategy**: For tensor and 2D parallelism (advanced distributed training).
- **FP8 + FSDP2 + torch.compile**: Support for new PyTorch features and precision.
- **New logger and callback options**: e.g., `checkpoint_path_prefix` for MLflow, improved CometML/Neptune support.
- **Improved fault tolerance**: More robust checkpointing and resuming.

---

### **How to List Updated Functions with context7**

If you want to use context7 to programmatically list updated functions, you would:
- Point context7 at the PyTorch Lightning repo or changelog.
- Query for "new functions", "updated API", or "breaking changes" in the latest release notes or API docs.

---

**Summary Table:**

| Area                | New/Updated Functionality                                 |
|---------------------|----------------------------------------------------------|
| Model API           | `configure_model`, manual optimization, `CombinedLoader` |
| Trainer API         | `barebones`, new checkpoint/resume, auto device select   |
| Precision/Dist      | New precision names, native FSDP, no DP                  |
| Logging/Callbacks   | `log_dict`, on_exception checkpoint, explicit loss log   |
| Removed/Changed     | `*_epoch_end`, `*_step_end`, `tune`, `track_grad_norm`   |

---

**References:**
- [PyTorch Lightning 2.5 Release Notes](https://github.com/Lightning-AI/pytorch-lightning/releases)
- [LightningModule API Docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
- [Upgrade Guide](https://lightning.ai/docs/pytorch/stable/upgrade/2.0_0.html)

Let me know if you want a script or context7 query example for this!
