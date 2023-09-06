from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm
from scipy.optimize import linprog
import numpy as np
import torch
from typing import Callable, Optional, Sequence, Union

from ignite.metrics.gan.utils import _BaseInceptionMetric, InceptionModel
from ignite.metrics.gan.fid import fid_score


class WInD(_BaseInceptionMetric):
    ''' Calculates Wasserstein Inception Distance '''
    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        n_components: int = 10
    ) -> None:

        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires numpy to be installed.")

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scipy to be installed.")

        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._eps = 1e-6
        self.n_components = n_components

        super(WInD, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    def reset(self) -> None:
        self._real_samples = None
        self._generated_samples = None

        self._test_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._num_examples: int = 0

        super(WInD, self).reset()


    def update(self, output: Sequence[torch.Tensor]) -> None:
        train, test = output
        real_features = self._extract_features(train)
        generated_features = self._extract_features(test)

        if real_features.shape[0] != generated_features.shape[0] or real_features.shape[1] != generated_features.shape[1]:
            raise ValueError(
                f"""
    Number of Training Features and Testing Features should be equal ({real_features.shape} != {generated_features.shape})
                """
            )

        if self._real_samples is None:
            # first batch
            self._real_samples = real_features
            self._generated_samples = generated_features
        else:
            # append to existing batches
            self._real_samples = torch.cat([self._real_samples, real_features], dim=0)
            self._generated_samples = torch.cat([self._generated_samples, generated_features], dim=0)


        self._num_examples += real_features.shape[0]
        

    def compute(self) -> float:
        # fit GMM on x and g (inception features of real and generated samples)
        gmm_x = GaussianMixture(n_components=self.n_components)
        gmm_g = GaussianMixture(n_components=self.n_components)
        gmm_x.fit(self._real_samples.cpu())
        gmm_g.fit(self._generated_samples.cpu())
        
        # compute fréchet distances between kernels
        mu_x, mu_g = torch.Tensor(gmm_x.means_), torch.Tensor(gmm_g.means_)
        sigma_x, sigma_g = torch.Tensor(gmm_x.covariances_), torch.Tensor(gmm_x.covariances_)
        pi_x, pi_g = gmm_x.weights_, gmm_g.weights_
    
        m, n = len(pi_x), len(pi_g)
    
        f_dist = np.zeros((m,n))
        for j in range(m):
            for k in range(n):
                f_dist[j,k] = fid_score(mu_x[j], mu_g[k], sigma_x[j], sigma_g[k])
    
        # create constraints
        sum_Wp = np.sum(pi_x)   
        sum_Wq = np.sum(pi_g) 
        movableEarth = np.min([sum_Wp, sum_Wq])
        f_dist = f_dist.flatten()
        
        A_ub = np.zeros((m+n,m*n))
        b_ub = np.zeros(m+ n)
        for i in range(m):
            for j in range(n):
                A_ub[i, i*n +j] = 1
                A_ub[j+m, (j) + i*n] = 1
        for i in range(m):
            b_ub[i] = pi_x[i]
        for i in range(m,m+ n):
            b_ub[i] = pi_g[i-m]
        A_eq = np.ones((1,m*n))
        b_eq = np.array([movableEarth])
        bounds = [(0,None) for i in range(m*n)]
    
        # optimize
        res = linprog(f_dist,  A_ub=A_ub, b_ub=b_ub, bounds=bounds, A_eq=A_eq, b_eq=b_eq)
        
        # compute wasserstein distance
        optimal_flow = res.x
        emd = np.sum(f_dist*optimal_flow)/np.sum(optimal_flow)
    
        return emd

