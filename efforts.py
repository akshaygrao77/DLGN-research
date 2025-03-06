import wandb
import tqdm

# Replace with your W&B username
USERNAME = "akshaygraocse"  

# Initialize API
api = wandb.Api()

# Fetch all projects
projects = api.projects(USERNAME)

total_runs = 0
total_gpu_hours = 0
total_cpu_hours = 0
total_runtime = 0  # in seconds
total_sweeps = 0
total_models_logged = 0

for i,project in enumerate(projects):
    print(f"Fetching data for project: {project.name}")
    runs = api.runs(f"{USERNAME}/{project.name}")
    
    total_runs += len(runs)
    loader = tqdm.tqdm(runs, desc='Analysing runs')
    for batch_idx, run in enumerate(loader, 0):
        runtime = run.summary.get("_runtime", 0)  # Runtime in seconds
        total_runtime += runtime

        # Extract GPU and CPU hours from system metrics
        history = run.history(keys=["_wandb"])
        if len(history) > 0 and "_wandb" in history.columns:
            gpu_hours = history["_wandb"].apply(lambda x: x.get("gpu_hours", 0)).sum()
            cpu_hours = history["_wandb"].apply(lambda x: x.get("cpu_hours", 0)).sum()
            total_gpu_hours += gpu_hours
            total_cpu_hours += cpu_hours

        # Count number of checkpoints/models logged
        total_models_logged += sum(1 for _ in run.files())  # Fixed: Count files

        # Check if part of a sweep
        if getattr(run, "sweep_id", None):
            total_sweeps += 1

# Convert runtime to hours
total_runtime_hours = total_runtime / 3600  

print("\n===== W&B Profile Stats =====")
print(f"Total Runs: {total_runs}")
print(f"Total GPU Hours: {total_gpu_hours:.2f}")
print(f"Total CPU Hours: {total_cpu_hours:.2f}")
print(f"Total Runtime (Hours): {total_runtime_hours:.2f}")
print(f"Total Sweeps Conducted: {total_sweeps}")
print(f"Total Models/Checkpoints Logged: {total_models_logged}")
