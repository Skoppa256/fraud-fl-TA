"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Any, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from experiments.determinism import derive_seed, torch_deterministic
from hfedxgboost.models import CNN, fit_xgboost
from hfedxgboost.utils import single_tree_preds_from_each_client


class FlClient(fl.client.Client):
    """Custom class contains the methods that the client need."""

    def __init__(
        self,
        cfg: DictConfig,
        trainloader: DataLoader,
        valloader: DataLoader,
        cid: str,
    ):
        self.cid = cid
        self.config = cfg

        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.valloader: Any

        # instantiate model
        self.net = CNN(cfg)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_one_loop(self, data, optimizer, metric_fn, criterion):
        """Trains the neural network model for one loop iteration.

        Parameters
        ----------
            data (tuple): A tuple containing the inputs and
            labels for the training data, where the input represent the predictions
            of the trees from the tree ensemples of the clients.

        Returns
        -------
            loss (float): The value of the loss function after the iteration.
            metric_val * n_samples (float): The value of the chosen evaluation metric
            (accuracy or MSE) after the iteration.
            n_samples (int): The number of samples used for training in the iteration.
        """
        tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
        optimizer.zero_grad()

        outputs = self.net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        n_samples = labels.size(0)
        # metric_fn (torchmetrics) is instantiated on CPU; outputs/labels live on
        # self.device (cuda when a GPU is present). Move both inputs to CPU to
        # match the metric's device — same pattern as the eval path in utils.py.
        # (Feeding cuda tensors to the CPU metric raises a device-mismatch error.)
        metric_val = metric_fn(outputs.cpu(), labels.type(torch.int).cpu())

        return loss.item(), metric_val * n_samples, n_samples

    def train(
        self,
        net: CNN,
        trainloader: DataLoader,
        num_iterations: int,
    ) -> Tuple[float, float, int]:
        """Train CNN model on a given dataset(trainloader) for(num_iterations).

        Parameters
        ----------
            net (CNN): The convolutional neural network to be trained.
            trainloader (DataLoader): The data loader object containing
             the training dataset.
            num_iterations (int): The number of iterations or batches to
             be processed by the network.

        Returns
        -------
            Tuple[float, float, int]: A tuple containing the average loss per sample,
            the average evaluation result per sample, and the total number of training
            samples processed.

        Note:

            - The training is formulated in terms of the number of updates or iterations
            processed by the network.
        """
        net.train()
        total_loss, total_result, total_n_samples = 0.0, 0.0, 0

        # Unusually, this training is formulated in terms of number of
        # updates/iterations/batches processed
        # by the network. This will be helpful later on, when partitioning the
        # data across clients: resulting
        # in differences between dataset sizes and hence inconsistent numbers of updates
        # per 'epoch'.
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.config.clients.CNN.lr, betas=(0.5, 0.999)
        )
        metric_fn = instantiate(self.config.dataset.task.metric.fn)
        criterion = instantiate(self.config.dataset.task.criterion)
        for _i, data in zip(range(num_iterations), trainloader):
            loss, metric_val, n_samples = self.train_one_loop(
                data, optimizer, metric_fn, criterion
            )
            total_loss += loss
            total_result += metric_val
            total_n_samples += n_samples

        return (
            total_loss / total_n_samples,
            total_result / total_n_samples,
            total_n_samples,
        )

    def test(self, net: CNN, testloader: DataLoader) -> Tuple[float, float, int]:
        """Evaluates the network on test data.

        Parameters
        ----------
         net: The CNN model to be tested.
         testloader: The data loader containing the test data.

         Return: A tuple containing the average loss,
         average metric result,
         and the total number of samples tested.
        """
        total_loss, total_result, n_samples = 0.0, 0.0, 0
        net.eval()
        metric_fn = instantiate(self.config.dataset.task.metric.fn)
        criterion = instantiate(self.config.dataset.task.criterion)
        with torch.no_grad():
            for data in testloader:
                tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(tree_outputs)
                total_loss += criterion(outputs, labels).item()
                n_samples += labels.size(0)
                metric_val = metric_fn(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += metric_val * labels.size(0)

        return total_loss / n_samples, total_result / n_samples, n_samples

    def _apply_local_oversampling(self, x_tensor, y_tensor, method: str):
        """Apply SMOTE or ADASYN locally before XGBoost training.

        ``method`` is one of {"smote", "adasyn", "none"}. Guards: skip if
        n_fraud < k_neighbors + 1 (default: < 6). ADASYN ValueErrors are
        swallowed by ``apply_adasyn`` and the original tensors are returned.
        """
        import sys
        import os
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..')
        )
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from omegaconf import OmegaConf
        from preprocessing.smote import apply_smote
        from preprocessing.adasyn import apply_adasyn

        x_np = x_tensor.detach().cpu().numpy().astype(np.float32)
        y_np = y_tensor.detach().cpu().numpy().astype(np.int32)

        client_dict = {
            'x': x_np,
            'y': y_np,
            'client_id': int(self.cid),
            'n_samples': len(y_np),
            'n_fraud': int(y_np.sum()),
            'fraud_ratio': float(y_np.mean()),
        }

        k_neighbors = OmegaConf.select(
            self.config, "dataset.oversampling.k_neighbors", default=5
        )
        sampling_strategy = OmegaConf.select(
            self.config, "dataset.oversampling.sampling_strategy", default="auto"
        )
        base_seed = OmegaConf.select(
            self.config, "dataset.oversampling.base_seed", default=42
        )

        if method == "smote":
            result = apply_smote(
                client_dict,
                enabled=True,
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy,
                base_seed=base_seed,
            )
        elif method == "adasyn":
            result = apply_adasyn(
                client_dict,
                enabled=True,
                n_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy,
                base_seed=base_seed,
            )
        else:
            return x_tensor, y_tensor

        x_out = torch.from_numpy(result['x'].astype(np.float32))
        y_out = torch.from_numpy(result['y'].astype(np.float32))
        return x_out, y_out

    def get_parameters(self, ins):
        """Round 0 of Ma et al. (2023), Algorithm 1: build this client's tree ensemble.

        Called once, by the server, before any CNN training; the ensemble is then
        fixed for rounds 1..R (§3.4) and ``fit`` never calls this again. The rows
        come in partition order (main.py builds an unshuffled loader) and XGBoost's
        ``random_state`` is the run seed, so the trees are the same in every run.

        Parameters
        ----------
        - self (object): The instance of the class that the function belongs to.
        - ins (GetParametersIns): An input parameter object.

        Returns
        -------
        Tuple[GetParametersRes,
        Union[Tuple[XGBClassifier, int],Tuple[XGBRegressor, int]]]:
            A tuple containing the parameters of the net and the tree.
            - GetParametersRes:
                - status : An object with the status code.
                - parameters : An ndarray containing the model's weights.
            - Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]:
                A tuple containing either an XGBClassifier or XGBRegressor
                object along with client's id.
        """
        for dataset in self.trainloader_original:
            data, label = dataset[0], dataset[1]

        from omegaconf import OmegaConf
        oversampling_method = OmegaConf.select(
            self.config, "dataset.oversampling.method", default="none"
        )
        if oversampling_method in ("smote", "adasyn"):
            data, label = self._apply_local_oversampling(
                data, label, oversampling_method
            )

        tree = fit_xgboost(
            self.config,
            self.config.dataset.task.task_type,
            data,
            label,
            self.config.n_estimators_client,
        )
        return GetParametersRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(self.net.get_weights()),
        ), (tree, int(self.cid))

    def fit(self, ins: FitIns) -> FitRes:
        """One of rounds 1..R (Ma et al. 2023, Algorithm 1 lines 8-13).

        Trains the CNN on the outputs of the fixed round-0 ensemble (all K·M trees)
        for this client's rows, and returns the CNN with this client's own trees,
        unchanged: the ensemble is fixed after round 0 (§3.4), so nothing here
        re-trains a tree. The CNN's budget is the seed-42 sweep's, not the paper's
        E epochs: ``num_iterations`` batches per round, the first ones of a shuffle
        drawn from (seed, round, client).
        """
        torch_deterministic()
        num_iterations = int(ins.config["num_iterations"])
        batch_size = int(ins.config["batch_size"])
        server_round = int(ins.config["round"])
        client_num = int(self.config.clients.client_num)

        self.net.set_weights(parameters_to_ndarrays(ins.parameters[0]))  # type: ignore # noqa: E501 # pylint: disable=line-too-long
        aggregated_trees = ins.parameters[1]  # type: ignore # noqa: E501 # pylint: disable=line-too-long
        if not isinstance(aggregated_trees, list) or len(aggregated_trees) != client_num:
            raise RuntimeError(
                f"Client {self.cid}: expected the round-0 ensemble of all "
                f"{client_num} clients' trees, got {type(aggregated_trees).__name__}"
                + (f" of {len(aggregated_trees)}" if isinstance(aggregated_trees, list) else "")
            )
        own_trees = [t for t in aggregated_trees if int(t[1]) == int(self.cid)]
        if len(own_trees) != 1:
            raise RuntimeError(
                f"Client {self.cid}: found {len(own_trees)} copies of its own trees "
                "in the received ensemble, expected 1"
            )

        # S*_k: the outputs of all K·M trees on this client's (pre-SMOTE) rows.
        tree_loader: Any = single_tree_preds_from_each_client(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.config.n_estimators_client,
            client_num,
        )
        trainloader = DataLoader(
            tree_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            generator=torch.Generator().manual_seed(
                derive_seed(int(self.config.random_seed), server_round, int(self.cid))
            ),
        )
        self.valloader = single_tree_preds_from_each_client(
            self.valloader_original,
            batch_size,
            aggregated_trees,
            self.config.n_estimators_client,
            client_num,
        )
        num_iterations = num_iterations or len(trainloader)
        print(
            "Client", self.cid, ": round", server_round, "training for",
            num_iterations, "iterations/updates",
        )
        self.net.to(self.device)
        train_loss, train_result, num_update_samples = self.train(
            self.net,
            trainloader,
            num_iterations=num_iterations,
        )
        # FedAvg weight N_k (Ma et al. 2023, Algorithm 1 line 13): the client's local
        # sample count — the PRE-SMOTE partition size, so oversampling does not
        # inflate a client's weight and Dirichlet heterogeneity is preserved.
        local_num_examples = len(self.trainloader_original.dataset)
        print(
            f"Client {self.cid}: training round complete, {num_update_samples}",
            f"update-samples over {num_iterations} updates; reporting FedAvg",
            f"weight N_k={local_num_examples} (pre-SMOTE local partition size)",
        )
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=(  # type: ignore # the hfedxgboost (CNN, (trees, cid)) pair
                GetParametersRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters(self.net.get_weights()),
                ),
                own_trees[0],
            ),
            num_examples=local_num_examples,
            metrics={
                "loss": train_loss,
                self.config.dataset.task.metric.name: train_result,
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate CNN model using the given evaluation parameters.

        Parameters
        ----------
          ins: An instance of EvaluateIns class that contains the parameters
          for evaluation.
        Return:
          An EvaluateRes object that contains the evaluation results.
        """
        # set the weights of the CNN net
        self.net.set_weights(parameters_to_ndarrays(ins.parameters))

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = self.test(
            self.net,
            self.valloader,
        )

        # Return evaluation information
        print(
            f"Client {self.cid}: evaluation on {num_examples} examples:",
            f"loss={loss:.4f}",
            self.config.dataset.task.metric.name,
            f"={result:.4f}",
        )
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss,
            num_examples=num_examples,
            metrics={self.config.dataset.task.metric.name: result},
        )
