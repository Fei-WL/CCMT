
import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import FairseqCriterion, register_criterion, LabelSmoothedCrossEntropyCriterionConfig, label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import logging

logger = logging.getLogger(__name__)

# def label_smoothed_nll_loss(_1st_lprobs, _2nd_lprobs, target, epsilon, ignore_index=None, reduce=True):
#     if target.dim() == _1st_lprobs.dim() - 1:
#         target = target.unsqueeze(-1)
#     _1st_nll_loss = -_1st_lprobs.gather(dim=-1, index=target)
#     _1st_smooth_loss = -_1st_lprobs.sum(dim=-1, keepdim=True)
#     _2nd_nll_loss = -_2nd_lprobs.gather(dim=-1, index=target)
#     _2nd_smooth_loss = -_2nd_lprobs.sum(dim=-1, keepdim=True)
#     if ignore_index is not None:
#         pad_mask = target.eq(ignore_index)
#         _1st_nll_loss.masked_fill_(pad_mask, 0.0)
#         _1st_smooth_loss.masked_fill_(pad_mask, 0.0)
#         _2nd_nll_loss.masked_fill_(pad_mask, 0.0)
#         _2nd_smooth_loss.masked_fill_(pad_mask, 0.0)
#     else:
#         _1st_nll_loss = _1st_nll_loss.squeeze(-1)
#         _1st_smooth_loss = _1st_smooth_loss.squeeze(-1)
#         _2nd_nll_loss = _2nd_nll_loss.squeeze(-1)
#         _2nd_smooth_loss = _2nd_smooth_loss.squeeze(-1)
#     if reduce:
#         _1st_nll_loss = _1st_nll_loss.sum()
#         _1st_smooth_loss = _1st_smooth_loss.sum()
#         _2nd_nll_loss = _2nd_nll_loss.sum()
#         _2nd_smooth_loss = _2nd_smooth_loss.sum()
#     _1st_eps_i = epsilon / (_1st_lprobs.size(-1) - 1)
#     _2nd_eps_i = epsilon / (_2nd_lprobs.size(-1) - 1)
#     loss = 0.5 * ((1.0 - epsilon - _1st_eps_i) * _1st_nll_loss + _1st_eps_i * _1st_smooth_loss) + \
#            0.5 * ((1.0 - epsilon - _2nd_eps_i) * _2nd_nll_loss + _2nd_eps_i * _2nd_smooth_loss)
#     nll_loss = 0.5 * _1st_nll_loss + 0.5 * _2nd_nll_loss
#     return loss, nll_loss


@register_criterion(
    "deliberate_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class deliberate_loss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        _2nd_lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        temp_net_output = (net_output[1]["dec_out"], net_output[1])
        # logger.info("Info of net_output:{}".format(net_output[0].size()))
        # logger.info("Info of temp_net_output:{}".format(temp_net_output[0].size()))
        _1st_lprobs, _ = self.get_lprobs_and_target(model, temp_net_output, sample)
        # logger.info("shape of _1st_lprobs:{}, shape of _2nd_lprobs:{}".format(_1st_lprobs.size(), _2nd_lprobs.size()))
        _2nd_loss, _2nd_nll_loss = label_smoothed_nll_loss(
            _2nd_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        _1st_loss, _1st_nll_loss = label_smoothed_nll_loss(
            _1st_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        loss = 0.5 * _1st_loss + 0.5 * _2nd_loss
        nll_loss = 0.5 * _1st_nll_loss + 0.5 * _2nd_nll_loss
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True