# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from math import log

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa


@register_criterion("glat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument('--mse-lambda', default=10, type=float, metavar='D')

    def _compute_oaxe_loss(self,  outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        if targets.dim() == 1:
            logits = F.log_softmax(outputs, dim=-1)
            losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")
            nll_loss = losses.float().mean().type_as(losses)
        else:

            bs, seq_len = list(targets.size())
            mask_ind = masks.unsqueeze(-1)
            # Bipart Loss
        
            target = targets.repeat(1, seq_len).view(bs, seq_len, seq_len)
            bipart_no_pad = target.ne(self.padding_idx)

            bipart_lprobs = F.log_softmax(outputs.float(), dim=-1)
                    
            nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs seq seq
            nll_loss = nll_loss * bipart_no_pad
            

            smooth_lprobs = F.log_softmax(outputs.float(), dim=-1)

            smooth_lprobs = smooth_lprobs.view(-1, bipart_lprobs.size(-1))
            smooth_loss = -smooth_lprobs.sum(dim=-1, keepdim=True)
            smooth_non_pad_mask = targets.view(-1, 1).ne(self.padding_idx)
            smooth_loss = smooth_loss * smooth_non_pad_mask
                

            best_match = np.repeat(np.arange(seq_len).reshape(1, -1, 1), bs, axis=0)# np.zeros((bs, seq_len, 1))
            nll_loss_numpy = nll_loss.detach().cpu().numpy()

            for batch_id in range(bs):
                no_pad_num = bipart_no_pad[batch_id, 0].sum()
                raw_index, col_index = lsa(nll_loss_numpy[batch_id, :no_pad_num, :no_pad_num])
                best_match[batch_id, :no_pad_num] = col_index.reshape(-1, 1)

            best_match = torch.Tensor(best_match).to(target).long()
            nll_loss = nll_loss * mask_ind

            nll_loss = nll_loss.gather(dim=-1, index=best_match)
            nll_loss = nll_loss.squeeze(-1)
            
            epsilon = 0.1
            eps_i = epsilon / bipart_lprobs.size(-1)
            nll_loss = (1 - epsilon) * nll_loss.float().mean().type_as(nll_loss) + eps_i * smooth_loss.float().mean().type_as(smooth_loss)
        
        loss = nll_loss
        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}


    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens, tgt_types = sample["target"], sample["prev_target"], sample["target_type"]
        if 'glat' in sample:
            glat = sample['glat']
        else:
            glat = None

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, tgt_types, glat)
        losses, nll_loss = [], []

        for obj in outputs:
            if obj.startswith('glat'):
                continue
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if "glat_accu" in outputs:
            logging_output["glat_accu"] = outputs['glat_accu']
        if "glat_context_p" in outputs:
            logging_output['glat_context_p'] = outputs['glat_context_p']

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        log_metric("glat_accu", logging_outputs)
        log_metric("glat_context_p", logging_outputs)

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def log_metric(key, logging_outputs):
    if len(logging_outputs) > 0 and key in logging_outputs[0]:
        metrics.log_scalar(
            key, utils.item(np.mean([log.get(key, 0) for log in logging_outputs])), priority=10, round=3
        )