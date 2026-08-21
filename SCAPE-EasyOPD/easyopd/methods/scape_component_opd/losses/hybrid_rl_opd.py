from __future__ import annotations


def hybrid_rl_opd(loss_rl: float, loss_opd: float, *, lambda_rl: float = 1.0, lambda_opd: float = 1.0) -> dict[str, float]:
    total = lambda_rl * float(loss_rl) + lambda_opd * float(loss_opd)
    return {"L_rl": float(loss_rl), "L_opd": float(loss_opd), "total_loss": total, "lambda_rl": lambda_rl, "lambda_opd": lambda_opd}
