import torch
import uuid
import wandb
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fine_tuning.logger_config import logger

from fine_tuning.utils import get_split_dataset


def load_model(config):
    """Load and configure the model with optional quantization"""
    logger.info(f"Loading model: {config.model_name}")

    # Configure quantization if using QLoRA
    if config.use_qlora:
        logger.info("Using QLoRA with 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, device_map="auto", dtype=torch.bfloat16
        )
        logger.info("Using full precision model. Training with gradient checkpointing.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model


def run_fine_tuning(config):
    """Run fine-tuning using centralized configuration"""
    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Generate unique run identifier
    run_uuid = str(uuid.uuid4())[:4]
    RUN_NAME = f"{config.model_name}-{config.graph_name}-{run_uuid}"

    # Setup directories
    train_trajectories_dir = config.train_trajectories_dir
    val_trajectories_dir = config.val_trajectories_dir

    # Extract model name from full model path (e.g., "Qwen/Qwen3-8B" -> "Qwen3-8B")
    model_name = config.model_name.split("/")[1]
    output_dir = config.output_dir / config.graph_name / model_name / run_uuid
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fine-tuning run: {RUN_NAME}")
    logger.info(f"Train trajectories directory: {train_trajectories_dir}")
    logger.info(f"Validation trajectories directory: {val_trajectories_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=RUN_NAME,
        dir=str(output_dir),
        config={
            "model_name": config.model_name,
            "graph_name": config.graph_name,
            "run_uuid": run_uuid,
            "output_dir": str(output_dir),
            "use_qlora": config.use_qlora,
            "rejection_sampling": config.rejection_sampling,
            "max_length": config.training.max_length,
        },
    )

    # Load model and tokenizer
    model = load_model(config)

    # Configure model for training
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load chat template
    chat_template_path = Path(__file__).parent / "chat_template.jinja"
    tokenizer.chat_template = chat_template_path.read_text()

    # Load and prepare dataset
    logger.info("Loading trajectory datasets")

    train_dataset = get_split_dataset("train", config)
    eval_dataset = get_split_dataset("val", config)

    logger.info(f"Loaded {len(train_dataset)} training and {len(eval_dataset)} validation samples.")

    # Log dataset info to wandb
    wandb.config.update(
        {
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset),
            "train_queries_limit": config.train_queries_limit,
            "val_queries_limit": config.val_queries_limit,
        }
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_length=config.training.max_length,
            packing=config.training.packing,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            eval_accumulation_steps=config.training.eval_accumulation_steps,
            num_train_epochs=config.training.num_train_epochs,
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            logging_steps=config.training.logging_steps,
            eval_strategy="steps",
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            save_strategy="steps",
            load_best_model_at_end=config.training.load_best_model_at_end,
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=config.training.greater_is_better,
            optim=config.training.optim,
            max_grad_norm=config.training.max_grad_norm,
            report_to="wandb",
            assistant_only_loss=config.training.assistant_only_loss,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            target_modules=config.lora.target_modules,
        ),
    )

    # Log parameters after PEFT is applied by trainer
    train_p, tot_p = trainer.model.get_nb_trainable_parameters()  # type: ignore
    logger.info(f"Trainable parameters:      {train_p/1e6:.2f}M")
    logger.info(f"Total parameters:          {tot_p/1e6:.2f}M")
    logger.info(f"% of trainable parameters: {100*train_p/tot_p:.2f}%")

    # Log model parameters to wandb
    wandb.config.update(
        {
            "trainable_parameters": train_p,
            "total_parameters": tot_p,
            "trainable_percentage": 100 * train_p / tot_p,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
        }
    )

    # Sanity checks before training
    logger.info("Running sanity checks on training data")
    dl = trainer.get_train_dataloader()
    cut_count = 0
    for i, batch in enumerate(dl):
        sup = int((batch["labels"] != -100).sum())
        if sup == 0:
            logger.error(f"Found batch {i} with no supervised tokens!")
        if batch["input_ids"].shape[1] == trainer.args.max_length:  # type: ignore
            cut_count += 1

    logger.warning(f"{cut_count} of {i+1} batches were cut to {trainer.args.max_length} tokens.")  # type: ignore

    # Start training
    logger.info("Starting fine-tuning")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Finish wandb run
    wandb.finish()

    logger.info(f"Fine-tuning completed successfully. Model saved to {output_dir}")

    return str(output_dir)
