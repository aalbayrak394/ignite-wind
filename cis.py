import torch
from typing import Callable, Optional, Union

from ignite.metrics.gan.utils import _BaseInceptionMetric, InceptionModel
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.exceptions import NotComputableError


class BCIS(_BaseInceptionMetric):
    _state_dict_all_req_keys = ("_num_examples", "_prob_total", "_total_kl_d")

    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._eps = 1e-16

        super(BCIS, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )


    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0

        self._prob_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._total_kl_d = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)

        super(BCIS, self).reset()


    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:
        x, labels = output

        # p(y|c) = E_x_c(p(y|x))
        separated_x = {
            label: x[(labels == torch.full_like(labels, label)).flatten()]
            for label in range(4)
        }
        probabilities = torch.stack([self._extract_features(separated_x[l]) for l in range(4)])
        probabilities = torch.mean(probabilities, dim=0)

        prob_sum = torch.sum(probabilities, 0, dtype=torch.float64)
        log_prob = torch.log(probabilities + self._eps)
        if log_prob.dtype != probabilities.dtype:
            log_prob = log_prob.to(probabilities)
        kl_sum = torch.sum(probabilities * log_prob, 0, dtype=torch.float64)

        self._num_examples += probabilities.shape[0]
        self._prob_total += prob_sum
        self._total_kl_d += kl_sum


    @sync_all_reduce("_num_examples", "_prob_total", "_total_kl_d")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("BCIS must have at least one example before it can be computed.")

        mean_probs = self._prob_total / self._num_examples
        log_mean_probs = torch.log(mean_probs + self._eps)
        if log_mean_probs.dtype != self._prob_total.dtype:
            log_mean_probs = log_mean_probs.to(self._prob_total)
        excess_entropy = self._prob_total * log_mean_probs
        avg_kl_d = torch.sum(self._total_kl_d - excess_entropy) / self._num_examples

        # BCIS : Between Class Inception Score
        return torch.exp(avg_kl_d).item()
    

class WCIS(_BaseInceptionMetric):
    _state_dict_all_req_keys = ("_num_examples", "_prob_total", "_total_kl_d")

    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._eps = 1e-16

        super(WCIS, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0

        self._prob_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._c_prob_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._total_kl_d = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)

        super(WCIS, self).reset()

    
    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:
        x, labels = output

        probabilities = self._extract_features(x)

        separated_x = {
            label: x[(labels == torch.full_like(labels, label)).flatten()]
            for label in range(4)
        }
        c_probabilities = torch.stack([self._extract_features(separated_x[l]) for l in range(4)])
        c_probabilities = torch.mean(probabilities, dim=0)

        prob_sum = torch.sum(probabilities, 0, dtype=torch.float64)
        c_prob_sum = torch.sum(c_probabilities, 0, dtype=torch.float64)
        log_prob = torch.log(probabilities + self._eps)
        if log_prob.dtype != probabilities.dtype:
            log_prob = log_prob.to(probabilities)
        kl_sum = torch.sum(probabilities * log_prob, 0, dtype=torch.float64)

        self._num_examples += probabilities.shape[0]
        self._prob_total += prob_sum
        self._c_prob_total += c_prob_sum
        self._total_kl_d += kl_sum


    @sync_all_reduce("_num_examples", "_prob_total", "_total_kl_d")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("WCIS must have at least one example before it can be computed.")

        mean_probs = self._c_prob_total / self._num_examples
        log_mean_probs = torch.log(mean_probs + self._eps)
        if log_mean_probs.dtype != self._prob_total.dtype:
            log_mean_probs = log_mean_probs.to(self._prob_total)
        excess_entropy = self._prob_total * log_mean_probs
        avg_kl_d = torch.sum(self._total_kl_d - excess_entropy) / self._num_examples

        # WCIS : Within Class Inception Score
        return torch.exp(avg_kl_d).item()