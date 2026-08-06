"""
==========================================================
Lab 27 - Mixed Precision Training
==========================================================

Objectives:

1. Understand FP32, FP16 and BF16.
2. Understand automatic mixed precision (AMP).
3. Use torch.autocast().
4. Understand FP16 gradient underflow.
5. Use GradScaler.
6. Build a complete mixed-precision training loop.
7. Combine mixed precision with:
      - Label Smoothing (Lab 25)
      - Learning Rate Scheduling (Lab 26)
8. Compare FP32 and mixed-precision training.

Mixed Precision Pipeline:

             INPUT
               |
               v
        +--------------+
        |   autocast   |
        |              |
        |    Model     |
        |      |       |
        |      v       |
        |    logits    |
        |      |       |
        |      v       |
        |     loss     |
        +------+-------+
               |
               v
        scaler.scale(loss)
               |
               v
           backward()
               |
               v
       scaled gradients
               |
               v
      scaler.step(optimizer)
               |
               v
        scaler.update()
"""

import time

import torch
import torch.nn as nn


# ==========================================================
# Configuration
# ==========================================================

INPUT_DIM = 128
HIDDEN_DIM = 512
OUTPUT_DIM = 100

BATCH_SIZE = 64

EPOCHS = 5

LEARNING_RATE = 0.001


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================================
# 1. Simple Neural Network
# ==========================================================

class SimpleModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


# ==========================================================
# 2. Floating Point Demonstration
# ==========================================================

def floating_point_demo():

    print()
    print("=" * 60)
    print("1. FLOATING POINT TYPES")
    print("=" * 60)

    x_fp32 = torch.tensor(
        [
            1.123456789,
            0.000000123456789
        ],
        dtype=torch.float32
    )

    x_fp16 = x_fp32.to(
        torch.float16
    )

    x_bf16 = x_fp32.to(
        torch.bfloat16
    )

    print()
    print("FP32:")
    print(x_fp32)

    print()
    print("FP16:")
    print(x_fp16)

    print()
    print("BF16:")
    print(x_bf16)

    print()

    print(
        "FP32 element size:",
        x_fp32.element_size(),
        "bytes"
    )

    print(
        "FP16 element size:",
        x_fp16.element_size(),
        "bytes"
    )

    print(
        "BF16 element size:",
        x_bf16.element_size(),
        "bytes"
    )


# ==========================================================
# 3. Model Parameter Dtype
# ==========================================================

def model_dtype_demo():

    print()
    print("=" * 60)
    print("2. MODEL PARAMETER DTYPE")
    print("=" * 60)

    model = SimpleModel(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM
    ).to(device)

    parameter = next(
        model.parameters()
    )

    print()
    print(
        "Model parameter dtype:"
    )

    print(
        parameter.dtype
    )

    print()
    print(
        "Important:"
    )

    print(
        "Using autocast does NOT mean that "
        "all model parameters permanently "
        "become FP16."
    )


# ==========================================================
# 4. Autocast Demonstration
# ==========================================================

def autocast_demo():

    print()
    print("=" * 60)
    print("3. AUTOCAST")
    print("=" * 60)

    if device.type != "cuda":

        print()
        print(
            "CUDA is not available."
        )

        print(
            "Skipping CUDA autocast demo."
        )

        return

    model = SimpleModel(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM
    ).to(device)

    inputs = torch.randn(
        BATCH_SIZE,
        INPUT_DIM,
        device=device
    )

    # ------------------------------------------------------
    # Normal FP32
    # ------------------------------------------------------

    output_fp32 = model(
        inputs
    )

    print()
    print(
        "Normal output dtype:"
    )

    print(
        output_fp32.dtype
    )

    # ------------------------------------------------------
    # Automatic Mixed Precision
    # ------------------------------------------------------

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        output_amp = model(
            inputs
        )

    print()
    print(
        "Autocast output dtype:"
    )

    print(
        output_amp.dtype
    )

    # ------------------------------------------------------
    # Parameters remain FP32
    # ------------------------------------------------------

    parameter = next(
        model.parameters()
    )

    print()
    print(
        "Model parameter dtype:"
    )

    print(
        parameter.dtype
    )


# ==========================================================
# 5. Create Fake Dataset
# ==========================================================

def create_fake_batch():

    inputs = torch.randn(
        BATCH_SIZE,
        INPUT_DIM,
        device=device
    )

    targets = torch.randint(
        low=0,
        high=OUTPUT_DIM,
        size=(BATCH_SIZE,),
        device=device
    )

    return (
        inputs,
        targets
    )


# ==========================================================
# 6. Normal FP32 Training
# ==========================================================

def train_fp32():

    print()
    print("=" * 60)
    print("4. FP32 TRAINING")
    print("=" * 60)

    model = SimpleModel(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        total_loss = 0.0

        # Simulate 50 batches

        for batch in range(50):

            inputs, targets = (
                create_fake_batch()
            )

            # ----------------------------------------------
            # Reset gradients
            # ----------------------------------------------

            optimizer.zero_grad()

            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            logits = model(
                inputs
            )

            # ----------------------------------------------
            # Loss
            # ----------------------------------------------

            loss = criterion(
                logits,
                targets
            )

            # ----------------------------------------------
            # Backward
            # ----------------------------------------------

            loss.backward()

            # ----------------------------------------------
            # Parameter update
            # ----------------------------------------------

            optimizer.step()

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss / 50
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print(
        f"FP32 training time: "
        f"{elapsed:.2f} seconds"
    )

    return model


# ==========================================================
# 7. Mixed Precision Training
# ==========================================================

def train_mixed_precision():

    print()
    print("=" * 60)
    print("5. MIXED PRECISION TRAINING")
    print("=" * 60)

    if device.type != "cuda":

        print()
        print(
            "CUDA is not available."
        )

        print(
            "FP16 AMP training requires "
            "a CUDA GPU for this lab."
        )

        return None

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = SimpleModel(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM
    ).to(device)

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ------------------------------------------------------
    # Gradient Scaler
    # ------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    start_time = time.time()

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        total_loss = 0.0

        for batch in range(50):

            inputs, targets = (
                create_fake_batch()
            )

            # ----------------------------------------------
            # Reset gradients
            # ----------------------------------------------

            optimizer.zero_grad()

            # ----------------------------------------------
            # Mixed Precision Forward Pass
            # ----------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(
                    inputs
                )

                loss = criterion(
                    logits,
                    targets
                )

            # ----------------------------------------------
            # Scaled Backpropagation
            # ----------------------------------------------

            scaler.scale(
                loss
            ).backward()

            # ----------------------------------------------
            # Optimizer Step
            # ----------------------------------------------

            scaler.step(
                optimizer
            )

            # ----------------------------------------------
            # Update Scale
            # ----------------------------------------------

            scaler.update()

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss / 50
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Scale: {scaler.get_scale():.0f}"
        )

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print(
        f"Mixed precision training time: "
        f"{elapsed:.2f} seconds"
    )

    return model


# ==========================================================
# 8. Gradient Scaling Demonstration
# ==========================================================

def gradient_scaling_demo():

    print()
    print("=" * 60)
    print("6. GRADIENT SCALING")
    print("=" * 60)

    print(
        """
Suppose the original loss is:

    loss = 0.000001

During FP16 training, very small gradients
can underflow toward zero.

GradScaler conceptually performs:

    scaled_loss = loss * scale

Example:

    loss  = 0.000001
    scale = 65536

    scaled_loss = 0.065536

Then:

    scaled_loss.backward()

produces larger temporary gradients.

Before the optimizer updates the parameters,
the gradients are effectively brought back
to the correct scale.

PyTorch manages this automatically with:

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()
"""
    )


# ==========================================================
# 9. Transformer Training Function
# ==========================================================

def train_transformer_mixed_precision(
    model,
    dataloader_en,
    dataloader_fr,
    optimizer,
    criterion,
    epochs,
    scheduler=None
):
    """
    Generic mixed-precision training function for the
    Transformer created in previous labs.

    Expected model output:

        logits, _, _, _ = model(
            inputs_en,
            inputs_fr
        )
    """

    if device.type != "cuda":

        raise RuntimeError(
            "This FP16 mixed-precision example "
            "expects CUDA."
        )

    # ------------------------------------------------------
    # Gradient Scaler
    # ------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    # ------------------------------------------------------
    # Training Mode
    # ------------------------------------------------------

    model.train()

    global_step = 0

    # ------------------------------------------------------
    # Epoch Loop
    # ------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        total_loss = 0.0

        batch_count = 0

        # --------------------------------------------------
        # Batch Loop
        # --------------------------------------------------

        for (
            (inputs_en, targets_en),
            (inputs_fr, targets_fr)

        ) in zip(
            dataloader_en,
            dataloader_fr
        ):

            # ----------------------------------------------
            # Move tensors to GPU
            # ----------------------------------------------

            inputs_en = inputs_en.to(
                device
            )

            inputs_fr = inputs_fr.to(
                device
            )

            targets_fr = targets_fr.to(
                device
            )

            # ----------------------------------------------
            # Reset gradients
            # ----------------------------------------------

            optimizer.zero_grad()

            # ----------------------------------------------
            # Forward + Loss under autocast
            # ----------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits, _, _, _ = model(
                    inputs_en,
                    inputs_fr
                )

                # logits:
                #
                # [batch,
                #  sequence_length,
                #  vocabulary]

                logits = logits.reshape(
                    -1,
                    logits.size(-1)
                )

                targets = targets_fr.reshape(
                    -1
                )

                loss = criterion(
                    logits,
                    targets
                )

            # ----------------------------------------------
            # Scale Loss + Backward
            # ----------------------------------------------

            scaler.scale(
                loss
            ).backward()

            # ----------------------------------------------
            # Optimizer Step
            # ----------------------------------------------

            scaler.step(
                optimizer
            )

            # ----------------------------------------------
            # Update GradScaler
            # ----------------------------------------------

            scaler.update()

            # ----------------------------------------------
            # LR Scheduler
            # ----------------------------------------------

            if scheduler is not None:

                current_lr = (
                    scheduler.step()
                )

            else:

                current_lr = (
                    optimizer
                    .param_groups[0]["lr"]
                )

            # ----------------------------------------------
            # Statistics
            # ----------------------------------------------

            total_loss += (
                loss.item()
            )

            batch_count += 1

            global_step += 1

        # --------------------------------------------------
        # Average Loss
        # --------------------------------------------------

        average_loss = (
            total_loss
            /
            max(batch_count, 1)
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Loss: {average_loss:.4f} "
            f"LR: {current_lr:.8f} "
            f"Scale: {scaler.get_scale():.0f}"
        )

    return model


# ==========================================================
# 10. Full Integration Example
# ==========================================================

def transformer_integration_example():

    print()
    print("=" * 60)
    print("7. TRANSFORMER INTEGRATION")
    print("=" * 60)

    print(
        """
Your final training stack can now contain:

LAB 25
------

    criterion = LabelSmoothingLoss(
        smoothing=0.1
    )


LAB 26
------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0
    )

    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=EMBEDDING_DIM,
        warmup_steps=4000
    )


LAB 27
------

    scaler = torch.amp.GradScaler("cuda")


TRAINING
--------

    model.train()

    for epoch in range(EPOCHS):

        for batch in dataloader:

            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits, _, _, _ = model(
                    inputs_en,
                    inputs_fr
                )

                logits = logits.reshape(
                    -1,
                    logits.size(-1)
                )

                targets = targets_fr.reshape(-1)

                loss = criterion(
                    logits,
                    targets
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()
"""
    )


# ==========================================================
# 11. Checkpoint Example
# ==========================================================

def checkpoint_example():

    print()
    print("=" * 60)
    print("8. CHECKPOINT")
    print("=" * 60)

    print(
        """
When using mixed precision, you can also save
the GradScaler state.

SAVE
----

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "epoch":
                epoch
        },

        "models/transformer.pth"
    )


LOAD
----

    checkpoint = torch.load(
        "models/transformer.pth"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )
"""
    )


# ==========================================================
# 12. Main
# ==========================================================

def main():

    print("=" * 60)
    print("LAB 27 - MIXED PRECISION TRAINING")
    print("=" * 60)

    print()
    print(
        "PyTorch version:",
        torch.__version__
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------
    # FP32 / FP16 / BF16
    # ------------------------------------------------------

    floating_point_demo()

    # ------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------

    model_dtype_demo()

    # ------------------------------------------------------
    # Autocast
    # ------------------------------------------------------

    autocast_demo()

    # ------------------------------------------------------
    # Gradient Scaling
    # ------------------------------------------------------

    gradient_scaling_demo()

    # ------------------------------------------------------
    # FP32 Training
    # ------------------------------------------------------

    train_fp32()

    # ------------------------------------------------------
    # Mixed Precision Training
    # ------------------------------------------------------

    train_mixed_precision()

    # ------------------------------------------------------
    # Transformer Integration
    # ------------------------------------------------------

    transformer_integration_example()

    # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    checkpoint_example()

    print()
    print("=" * 60)
    print("LAB 27 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()