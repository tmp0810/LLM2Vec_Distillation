import torch
from .cross_entropy_loss import CrossEntropyLoss


class VariousDivergence(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super(VariousDivergence, self).__init__(args)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.tea_temp = args.teacher_temperature
        self.kd_objective = args.kd_objective
        self.args = args

        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True)
        teacher_logits = teacher_outputs.logits
        
        
        kd_loss = self.dist_func(logits, teacher_logits, output_data["labels"])
        log["kd_loss"] = kd_loss

        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_accuracy(logits, output_data["labels"])
        log["accuracy"] = accuracy

        if self.args.report_logits:
            self.record_logits(
                logits, 
                output_data["labels"], 
                log, 
                teacher_logits=teacher_logits, 
                teacher_target=output_data["labels"]
            )

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    def compute_forward_kl_divergence(
    self, 
    logits, 
    teacher_logits, 
    target, 
    reduction="sum", 
    log=None, 
    use_tea_temp=False
):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs)).sum(-1)
        
        if reduction == "sum":
            kld = kld.sum()
            if log is not None:
                log["forward_kl"] = kld
        elif reduction == "mean":
            kld = kld.mean()
            if log is not None:
                log["forward_kl"] = kld

        return kld
    
    def compute_reverse_kl_divergence(
    self, 
    logits, 
    teacher_logits, 
    target, 
    reduction="sum", 
    log=None, 
    use_tea_temp=False
):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (probs * (lprobs - teacher_lprobs)).sum(-1)
        
        if reduction == "sum":
            kld = kld.sum()
            if log is not None:
                log["reverse_kl"] = kld
        elif reduction == "mean":
            kld = kld.mean()
            if log is not None:
                log["reverse_kl"] = kld

        return kld
    
    def compute_adaptive_kl_divergence(
    self, 
    logits, 
    teacher_logits, 
    target, 
    reduction="sum", 
    log=None, 
    use_tea_temp=False
):
        alpha = self.args.adaptive_kl_alpha
        probs = torch.softmax(
            logits / self.kd_temp, dim=-1, dtype=torch.float32
        )
        if use_tea_temp:
            teacher_probs = torch.softmax(
                teacher_logits / self.tea_temp / self.kd_temp, dim=-1, dtype=torch.float32
            )
        else:
            teacher_probs = torch.softmax(
                teacher_logits / self.kd_temp, dim=-1, dtype=torch.float32
            )
        sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
        sorted_probs = probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
        tail_mask = cum_teacher_probs.le(alpha).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        fkl = self.compute_forward_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)
        rkl = self.compute_reverse_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl
        
        if reduction == "sum":
            akl = akl.sum()
            if log is not None:
                log["adaptive_kl"] = akl
        elif reduction == "mean":
            akl = akl.mean()
            if log is not None:
                log["adaptive_kl"] = akl

        return akl
    
    def compute_skewed_forward_kl_divergence(
    self, 
    logits, 
    teacher_logits, 
    target, 
    reduction="sum", 
    log=None, 
    use_tea_temp=False
):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - mixed_lprobs)).sum(-1)
        
        if reduction == "sum":
            kld = kld.sum()
            if log is not None:
                log["skewed_forward_kl"] = kld
        elif reduction == "mean":
            kld = kld.mean()
            if log is not None:
                log["skewed_forward_kl"] = kld

        return kld
    
    def compute_skewed_reverse_kl_divergence(
    self, 
    logits, 
    teacher_logits, 
    target, 
    reduction="sum", 
    log=None, 
    use_tea_temp=False
):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs)).sum(-1)
        
        if reduction == "sum":
            kld = kld.sum()
            if log is not None:
                log["skewed_reverse_kl"] = kld
        elif reduction == "mean":
            kld = kld.mean()
            if log is not None:
                log["skewed_reverse_kl"] = kld

        return kld

    def compute_js_divergence(
    self, 
    logits, 
    teacher_logits, 
    target, 
    reduction="sum", 
    log=None, 
    use_tea_temp=False
):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        m_probs = (probs + teacher_probs) / 2
        
        lprobs = torch.log(probs + 1e-9)
        teacher_lprobs = torch.log(teacher_probs + 1e-9)
        m_lprobs = torch.log(m_probs + 1e-9)

        kld1 = (teacher_probs * (teacher_lprobs - m_lprobs)).sum(-1)
        kld2 = (probs * (lprobs - m_lprobs)).sum(-1)
        kld = (kld1 + kld2) / 2
        
        if reduction == "sum":
            kld = kld.sum()
            if log is not None:
                log["js_div"] = kld
        elif reduction == "mean":
            kld = kld.mean()
            if log is not None:
                log["js_div"] = kld

        return kld
