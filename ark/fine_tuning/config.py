from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# Fine-tuning Configuration
class LoraConfig(BaseModel):
    r: int = Field(description="LoRA rank")
    lora_alpha: int = Field(description="LoRA alpha parameter")
    lora_dropout: float = Field(description="LoRA dropout rate")
    bias: str = Field(description="Bias type")
    task_type: str = Field(description="Task type")
    target_modules: List[str] = Field(description="Target modules for LoRA")


class TrainingConfig(BaseModel):
    max_length: Optional[int] = Field(description="Maximum sequence length")
    packing: bool = Field(description="Whether to pack sequences")
    per_device_train_batch_size: int = Field(description="Training batch size per device")
    per_device_eval_batch_size: int = Field(description="Evaluation batch size per device")
    gradient_accumulation_steps: int = Field(description="Gradient accumulation steps")
    eval_accumulation_steps: int = Field(description="Evaluation accumulation steps for memory efficiency")
    num_train_epochs: int = Field(description="Number of training epochs")
    learning_rate: float = Field(description="Learning rate")
    warmup_steps: int = Field(description="Warmup steps")
    weight_decay: float = Field(description="Weight decay")
    logging_steps: int = Field(description="Logging frequency")
    eval_steps: int = Field(description="Evaluation frequency")
    save_steps: int = Field(description="Save frequency")
    load_best_model_at_end: bool = Field(description="Load best model at end")
    metric_for_best_model: str = Field(description="Metric for best model selection")
    greater_is_better: bool = Field(description="Whether greater metric is better")
    optim: str = Field(description="Optimizer")
    max_grad_norm: float = Field(description="Max gradient norm")
    assistant_only_loss: bool = Field(description="Only compute loss on assistant tokens")


class FineTuningConfig(BaseModel):
    graph_name: str = Field(description="Graph for fine-tuning")
    model_name: str = Field(description="Base model name")
    train_trajectories_dir: Path = Field(description="Directory with training trajectories")
    val_trajectories_dir: Path = Field(description="Directory with validation trajectories")
    output_dir: Path = Field(description="Output directory for fine-tuned model")
    train_queries_limit: int = Field(description="Number of training queries to use")
    val_queries_limit: int = Field(description="Number of validation queries to use")
    wandb_project: str = Field(description="Weights & Biases project name")
    use_qlora: bool = Field(description="Use QLoRA (4-bit quantization) for memory efficiency")
    rejection_sampling: bool = Field(description="Use rejection sampling to filter training data")

    @field_validator('train_trajectories_dir', 'val_trajectories_dir', 'output_dir', mode='before')
    def convert_to_path(cls, v, info):
        if isinstance(v, str) and info.data:
            # Replace {graph_name} with the actual graph_name value
            v = v.format(graph_name=info.data.get('graph_name', ''))
        return Path(v) if isinstance(v, str) else v

    # Training configuration
    lora: LoraConfig = Field(description="LoRA configuration")
    training: TrainingConfig = Field(description="Training configuration")

