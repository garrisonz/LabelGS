import wandb

# Start a W&B Run with wandb.init
run = wandb.init(project="my_first_project")

# Save model inputs and hyperparameters in a wandb.config object
config = run.config
config.learning_rate = 0.01

# Model training code here ...

# Log metrics over time to visualize performance with wandb.log
for i in range(10):
    run.log({"loss": ...})

# Mark the run as finished, and finish uploading all data
run.finish()