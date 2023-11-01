import torch as t

from complexity import SGLDParams, sgld
from data import get_filtered_loader
from models import CNN, MLP, get_accuracy, train_model

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

SGLD_PARAMS = SGLDParams(
    gamma=10,
    epsilon=0.0002,
    n_steps=2000,
    m=256,
)

results = {"mlp": {}, "cnn": {}}


def run():
    for i in range(1, 10):
        task_loader, task_data = get_filtered_loader(max_label=i)
        test_loader, test_data = get_filtered_loader(max_label=i, training=False)

        mlp_model = MLP().to(DEVICE)
        cnn_model = CNN().to(DEVICE)

        train_model("mlp20", mlp_model, task_loader, max_label=i, epochs=20)
        print(f"MLP Accuracy: {get_accuracy(mlp_model, test_loader)}")

        train_model("cnn20", cnn_model, task_loader, max_label=i, epochs=20)
        print(f"CNN Accuracy: {get_accuracy(cnn_model, test_loader)}")

        _, mlp_lambda_hat = sgld(mlp_model, SGLD_PARAMS, task_data, max_label=i)
        _, cnn_lambda_hat = sgld(cnn_model, SGLD_PARAMS, task_data, max_label=i)

        results["mlp"][i] = mlp_lambda_hat
        results["cnn"][i] = cnn_lambda_hat

        return results


run()
