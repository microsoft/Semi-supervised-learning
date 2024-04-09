import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc


class AUAvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    """
    def __init__(self, loss_conf={'device': 'cuda'}):
        super(AUAvULoss, self).__init__()
        self.device = loss_conf['device']
        self.beta = loss_conf['avuc_beta']
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.eps = 1e-12
        self.result = {}

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def auc_avu(self, logits, labels, unc):
        """ returns AvU at various uncertainty thresholds"""
        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        avu_list = []
        unc_list = []

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        auc_avu = torch.ones(1, device=labels.device)
        auc_avu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t) * (umax - umin))
            n_ac = torch.zeros(
                1,
                device=labels.device)  # number of samples accurate and certain
            n_ic = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and certain
            n_au = torch.zeros(1, device=labels.device
                               )  # number of samples accurate and uncertain
            n_iu = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and uncertain

            for i in range(len(labels)):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + 1e-10)
            avu_list.append(AvU.data.cpu().numpy())
            unc_list.append(unc_th)

        auc_avu = auc(th_list, avu_list)
        return auc_avu

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        n_ac = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and certain
        n_ic = torch.zeros(1, device=true_label.device
                           )  # number of samples inaccurate and certain
        n_au = torch.zeros(1, device=true_label.device
                           )  # number of samples accurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device
                           )  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels, idx, type=0):
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        avu_list = []
        unc_list = []

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        auc_avu = torch.ones(1, device=labels.device)
        auc_avu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t, device=labels.device) *
                             (umax - umin))
            n_ac = torch.zeros(
                1,
                device=labels.device)  # number of samples accurate and certain
            n_ic = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and certain
            n_au = torch.zeros(1, device=labels.device
                               )  # number of samples accurate and uncertain
            n_iu = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and uncertain

            for i in range(len(labels)):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)

        auc_avu = auc(th_list, list(map(lambda x: x.item(), avu_list)))
        avu_loss = -1 * self.beta * torch.log(torch.Tensor([auc_avu + self.eps]).to(self.device))
        ce_loss = self.cross_entropy_loss(logits, labels)
        loss = avu_loss + ce_loss


        self.result['loss'] = loss.item()

        return loss

    def batch_closure_callback(self, batch_state):
        pass
    
    def epoch_closure_callback(self, epoch_state):
        pass