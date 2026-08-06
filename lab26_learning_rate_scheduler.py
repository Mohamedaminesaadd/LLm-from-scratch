"""
==========================================================
Lab 26 - Learning Rate Scheduling
==========================================================

Objectives:

1. Understand why the learning rate should change.
2. Explore StepLR.
3. Explore ExponentialLR.
4. Explore CosineAnnealingLR.
5. Understand warmup.
6. Implement the Transformer learning-rate schedule.
7. Change learning rate at every training step.
8. Integrate the scheduler with Adam.
9. Save and restore scheduler state.

Transformer schedule:

    lr = d_model^(-0.5) *
         min(
             step^(-0.5),
             step * warmup_steps^(-1.5)
         )

During warmup:
    learning rate increases.

After warmup:
    learning rate decreases approximately as:

        1 / sqrt(step)
"""

import math

import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

LEARNING_RATE = 0.001

EMBEDDING_DIM = 128

WARMUP_STEPS = 400

TOTAL_STEPS = 2000


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# Simple Model Used For Demonstrations
# =========================================================

class SimpleModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(
            10,
            5
        )

    def forward(self, x):

        return self.linear(x)


# =========================================================
# Helper Function
# =========================================================

def get_learning_rate(
    optimizer
):

    return optimizer.param_groups[0]["lr"]


# =========================================================
# 1. Fixed Learning Rate
# =========================================================

def fixed_learning_rate_example():

    print()
    print("=" * 60)
    print("1. FIXED LEARNING RATE")
    print("=" * 60)

    model = SimpleModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(1, 11):

        lr = get_learning_rate(
            optimizer
        )

        print(
            f"Epoch {epoch:2d} "
            f"LR = {lr:.8f}"
        )


# =========================================================
# 2. StepLR
# =========================================================

def step_lr_example():

    print()
    print("=" * 60)
    print("2. STEP LR")
    print("=" * 60)

    model = SimpleModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Every 3 epochs:
    #
    # LR = LR * 0.5
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    for epoch in range(1, 11):

        lr = get_learning_rate(
            optimizer
        )

        print(
            f"Epoch {epoch:2d} "
            f"LR = {lr:.8f}"
        )

        # Normally called after optimizer.step()
        scheduler.step()


# =========================================================
# 3. ExponentialLR
# =========================================================

def exponential_lr_example():

    print()
    print("=" * 60)
    print("3. EXPONENTIAL LR")
    print("=" * 60)

    model = SimpleModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Each epoch:
    #
    # new_lr = old_lr * gamma
    # -----------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.9
        )
    )

    for epoch in range(1, 11):

        lr = get_learning_rate(
            optimizer
        )

        print(
            f"Epoch {epoch:2d} "
            f"LR = {lr:.8f}"
        )

        scheduler.step()


# =========================================================
# 4. Cosine Annealing
# =========================================================

def cosine_lr_example():

    print()
    print("=" * 60)
    print("4. COSINE ANNEALING")
    print("=" * 60)

    model = SimpleModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=0.00001
        )
    )

    for epoch in range(1, 11):

        lr = get_learning_rate(
            optimizer
        )

        print(
            f"Epoch {epoch:2d} "
            f"LR = {lr:.8f}"
        )

        scheduler.step()


# =========================================================
# 5. Transformer Learning Rate Formula
# =========================================================

def transformer_learning_rate(
    step,
    d_model,
    warmup_steps
):
    """
    Original Transformer learning-rate schedule.

    lr = d_model^(-0.5)
         *
         min(
             step^(-0.5),
             step * warmup_steps^(-1.5)
         )
    """

    # step = 0 would cause:
    #
    # 0 ** -0.5
    #
    # which is invalid.

    step = max(
        step,
        1
    )

    first_term = (
        step ** -0.5
    )

    second_term = (
        step
        *
        warmup_steps ** -1.5
    )

    learning_rate = (
        d_model ** -0.5
        *
        min(
            first_term,
            second_term
        )
    )

    return learning_rate


# =========================================================
# 6. Explore Transformer Schedule
# =========================================================

def transformer_schedule_example():

    print()
    print("=" * 60)
    print("5. TRANSFORMER LR SCHEDULE")
    print("=" * 60)

    interesting_steps = [
        1,
        10,
        50,
        100,
        200,
        300,
        400,
        500,
        800,
        1000,
        1500,
        2000
    ]

    print()

    print(
        f"d_model      = {EMBEDDING_DIM}"
    )

    print(
        f"warmup_steps = {WARMUP_STEPS}"
    )

    print()

    for step in interesting_steps:

        lr = transformer_learning_rate(
            step=step,
            d_model=EMBEDDING_DIM,
            warmup_steps=WARMUP_STEPS
        )

        phase = (
            "WARMUP"
            if step <= WARMUP_STEPS
            else "DECAY"
        )

        print(
            f"Step {step:4d} | "
            f"{phase:6s} | "
            f"LR = {lr:.8f}"
        )


# =========================================================
# 7. Custom Transformer Scheduler
# =========================================================

class TransformerLRScheduler:
    """
    Learning-rate scheduler used for Transformer training.

    It modifies the learning rate stored inside the
    optimizer.

    Usage:

        optimizer = Adam(...)

        scheduler = TransformerLRScheduler(
            optimizer,
            d_model=128,
            warmup_steps=400
        )

        ...

        optimizer.step()
        scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        d_model,
        warmup_steps
    ):

        if d_model <= 0:

            raise ValueError(
                "d_model must be positive."
            )

        if warmup_steps <= 0:

            raise ValueError(
                "warmup_steps must be positive."
            )

        self.optimizer = optimizer

        self.d_model = d_model

        self.warmup_steps = warmup_steps

        self.step_number = 0

        self.current_lr = 0.0


    # -----------------------------------------------------
    # Calculate LR
    # -----------------------------------------------------

    def calculate_lr(
        self,
        step=None
    ):

        if step is None:

            step = self.step_number

        step = max(
            step,
            1
        )

        learning_rate = (
            self.d_model ** -0.5
            *
            min(
                step ** -0.5,

                step
                *
                self.warmup_steps ** -1.5
            )
        )

        return learning_rate


    # -----------------------------------------------------
    # Scheduler Step
    # -----------------------------------------------------

    def step(self):

        self.step_number += 1

        learning_rate = (
            self.calculate_lr(
                self.step_number
            )
        )

        # Change LR inside every optimizer parameter group

        for param_group in (
            self.optimizer.param_groups
        ):

            param_group["lr"] = (
                learning_rate
            )

        self.current_lr = learning_rate

        return learning_rate


    # -----------------------------------------------------
    # Get Current LR
    # -----------------------------------------------------

    def get_last_lr(self):

        return self.current_lr


    # -----------------------------------------------------
    # Save Scheduler State
    # -----------------------------------------------------

    def state_dict(self):

        return {
            "step_number":
                self.step_number,

            "d_model":
                self.d_model,

            "warmup_steps":
                self.warmup_steps,

            "current_lr":
                self.current_lr
        }


    # -----------------------------------------------------
    # Restore Scheduler State
    # -----------------------------------------------------

    def load_state_dict(
        self,
        state
    ):

        self.step_number = (
            state["step_number"]
        )

        self.d_model = (
            state["d_model"]
        )

        self.warmup_steps = (
            state["warmup_steps"]
        )

        self.current_lr = (
            state["current_lr"]
        )

        # Restore LR into optimizer

        for param_group in (
            self.optimizer.param_groups
        ):

            param_group["lr"] = (
                self.current_lr
            )


# =========================================================
# 8. Test Custom Scheduler
# =========================================================

def test_custom_scheduler():

    print()
    print("=" * 60)
    print("6. CUSTOM TRANSFORMER SCHEDULER")
    print("=" * 60)

    model = SimpleModel()

    # -----------------------------------------------------
    # The scheduler will control the LR.
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0
    )

    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=EMBEDDING_DIM,
        warmup_steps=WARMUP_STEPS
    )

    for step in range(
        1,
        TOTAL_STEPS + 1
    ):

        lr = scheduler.step()

        if (
            step <= 10
            or step % 100 == 0
        ):

            phase = (
                "WARMUP"
                if step <= WARMUP_STEPS
                else "DECAY"
            )

            print(
                f"Step {step:4d} | "
                f"{phase:6s} | "
                f"LR = {lr:.8f}"
            )


# =========================================================
# 9. Real Training Example
# =========================================================

def training_example():

    print()
    print("=" * 60)
    print("7. TRAINING EXAMPLE")
    print("=" * 60)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = SimpleModel().to(
        device
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=EMBEDDING_DIM,
        warmup_steps=20
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    epochs = 5

    batches_per_epoch = 10

    global_step = 0

    for epoch in range(
        1,
        epochs + 1
    ):

        total_loss = 0.0

        for batch in range(
            batches_per_epoch
        ):

            # ---------------------------------------------
            # Fake training data
            # ---------------------------------------------

            inputs = torch.randn(
                8,
                10,
                device=device
            )

            targets = torch.randint(
                low=0,
                high=5,
                size=(8,),
                device=device
            )

            # ---------------------------------------------
            # Reset gradients
            # ---------------------------------------------

            optimizer.zero_grad()

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            logits = model(
                inputs
            )

            # ---------------------------------------------
            # Loss
            # ---------------------------------------------

            loss = criterion(
                logits,
                targets
            )

            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            loss.backward()

            # ---------------------------------------------
            # Update parameters
            # ---------------------------------------------

            optimizer.step()

            # ---------------------------------------------
            # Update LR
            # ---------------------------------------------

            current_lr = (
                scheduler.step()
            )

            global_step += 1

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss
            /
            batches_per_epoch
        )

        print(
            f"Epoch {epoch:2d} | "
            f"Step {global_step:3d} | "
            f"Loss = {average_loss:.4f} | "
            f"LR = {current_lr:.8f}"
        )


# =========================================================
# 10. Transformer Training Integration
# =========================================================

def transformer_integration_example():

    print()
    print("=" * 60)
    print("8. LAB 19 INTEGRATION")
    print("=" * 60)

    print(
        """
In Lab 19 you had:

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


Instead use:

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0
    )

    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=EMBEDDING_DIM,
        warmup_steps=4000
    )


Then inside the training loop:

    for epoch in range(EPOCHS):

        for (
            (inputs_en, targets_en),
            (inputs_fr, targets_fr)

        ) in zip(
            dataloader_en,
            dataloader_fr
        ):

            inputs_en = inputs_en.to(device)
            inputs_fr = inputs_fr.to(device)
            targets_fr = targets_fr.to(device)

            # -------------------------------
            # Reset gradients
            # -------------------------------

            optimizer.zero_grad()

            # -------------------------------
            # Forward
            # -------------------------------

            logits, _, _, _ = model(
                inputs_en,
                inputs_fr
            )

            # -------------------------------
            # Flatten
            # -------------------------------

            logits = logits.reshape(
                -1,
                logits.size(-1)
            )

            targets = targets_fr.reshape(
                -1
            )

            # -------------------------------
            # Loss
            # -------------------------------

            loss = criterion(
                logits,
                targets
            )

            # -------------------------------
            # Backward
            # -------------------------------

            loss.backward()

            # -------------------------------
            # Update model
            # -------------------------------

            optimizer.step()

            # -------------------------------
            # Update learning rate
            # -------------------------------

            current_lr = scheduler.step()
"""
    )


# =========================================================
# 11. Save / Load Scheduler
# =========================================================

def checkpoint_example():

    print()
    print("=" * 60)
    print("9. CHECKPOINT EXAMPLE")
    print("=" * 60)

    print(
        """
When saving your Transformer:

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "epoch":
                epoch
        },

        "models/transformer.pth"
    )


When loading:

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
"""
    )


# =========================================================
# 12. Main
# =========================================================

def main():

    print("=" * 60)
    print("LAB 26 - LEARNING RATE SCHEDULING")
    print("=" * 60)

    print()
    print(
        "Initial learning rate:",
        LEARNING_RATE
    )

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    print(
        "Warmup steps:",
        WARMUP_STEPS
    )

    # -----------------------------------------------------
    # Fixed LR
    # -----------------------------------------------------

    fixed_learning_rate_example()

    # -----------------------------------------------------
    # PyTorch schedulers
    # -----------------------------------------------------

    step_lr_example()

    exponential_lr_example()

    cosine_lr_example()

    # -----------------------------------------------------
    # Transformer schedule
    # -----------------------------------------------------

    transformer_schedule_example()

    # -----------------------------------------------------
    # Custom scheduler
    # -----------------------------------------------------

    test_custom_scheduler()

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    training_example()

    # -----------------------------------------------------
    # Integration explanation
    # -----------------------------------------------------

    transformer_integration_example()

    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------

    checkpoint_example()

    print()
    print("=" * 60)
    print("LAB 26 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()