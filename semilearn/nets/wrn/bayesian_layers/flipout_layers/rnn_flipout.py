# Copyright (C) 2021 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# LSTM Flipout Layer with reparameterization estimator to perform
# variational inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
# required to compute Evidence Lower Bound (ELBO).
#
# @authors: Piero Esposito
#
# ======================================================================================
from .linear_flipout import LinearFlipout
from ..base_variational_layer import BaseVariationalLayer_

import torch


class LSTMFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements LSTM layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        self.posterior_rho_init = posterior_rho_init,  # variance of weight --> sigma = log (1 + exp(rho))
        self.bias = bias

        self.kl = 0

        self.ih = LinearFlipout(prior_mean=prior_mean,
                                prior_variance=prior_variance,
                                posterior_mu_init=posterior_mu_init,
                                posterior_rho_init=posterior_rho_init,
                                in_features=in_features,
                                out_features=out_features * 4,
                                bias=bias)

        self.hh = LinearFlipout(prior_mean=prior_mean,
                                prior_variance=prior_variance,
                                posterior_mu_init=posterior_mu_init,
                                posterior_rho_init=posterior_rho_init,
                                in_features=out_features,
                                out_features=out_features * 4,
                                bias=bias)

    def kl_loss(self):
        kl_i = self.ih.kl_loss()
        kl_h = self.hh.kl_loss()
        return kl_i + kl_h

    def forward(self, X, hidden_states=None, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        batch_size, seq_size, _ = X.size()

        hidden_seq = []
        c_ts = []

        if hidden_states is None:
            h_t, c_t = (torch.zeros(batch_size,
                                    self.out_features).to(X.device),
                        torch.zeros(batch_size,
                                    self.out_features).to(X.device))
        else:
            h_t, c_t = hidden_states

        HS = self.out_features
        kl = 0
        for t in range(seq_size):
            x_t = X[:, t, :]

            ff_i, kl_i = self.ih(x_t)
            ff_h, kl_h = self.hh(h_t)
            gates = ff_i + ff_h

            kl += kl_i + kl_h

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
            c_ts.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        c_ts = torch.cat(c_ts, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_ts = c_ts.transpose(0, 1).contiguous()

        self.kl = kl
        if return_kl:
            return hidden_seq, (hidden_seq, c_ts), kl
        return hidden_seq, (hidden_seq, c_ts)
