import mlflow
import random

# 1. Указываем MLflow, куда сохранять результаты.
#    Он создаст папку 'mlruns' в текущей директории.
mlflow.set_tracking_uri("file:./mlruns")

# 2. Назначаем имя "эксперимента" (группы запусков).
mlflow.set_experiment("My Test Experiment")

# 3. Начинаем новый запуск.
with mlflow.start_run(run_name="Simple Log Test"):
    print("MLflow run started...")

    # 4. Логируем параметры (то, что не меняется во время обучения).
    params = {
        "learning_rate": 0.01,
        "optimizer": "Adam"
    }
    mlflow.log_params(params)
    print(f"Logged params: {params}")

    # 5. Логируем метрики (то, что меняется, как будто идёт обучение).
    print("Logging metrics for 5 steps...")
    for step in range(5):
        metrics = {
            "loss": 1 / (step + 1),
            "accuracy": random.uniform(0.7, 0.9)
        }
        mlflow.log_metrics(metrics, step=step)
        print(f"Step {step}: Logged metrics: {metrics}")

print("\nMLflow run finished successfully!")
print("Run 'mlflow ui' in your terminal and open http://localhost:5000 to see the results.")