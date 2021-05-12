import torch


def normalize_single_block(importance):
    """
    0-1 normalization over all parameters
    :param importance: [ (k1, p1), ..., (kn, pn)] (key, parameter) list
    """

    max_imp = -1
    min_imp = 1e7
    for _, imp in importance:
        curr_max_imp, curr_min_imp = imp.max(), imp.min()
        max_imp = max(max_imp, curr_max_imp)
        min_imp = min(min_imp, curr_min_imp)

    r = max(max_imp - min_imp, 1e-6)
    for _, imp in importance:
        imp -= min_imp
        imp /= float(r)

    return importance

def normalize_blocks(importance):
    """
    0-1 normalization over each parameter block
    :param importance: [ (k1, p1), ..., (kn, pn)] (key, parameter) list
    """

    for _, imp in importance:
        max_imp, min_imp = imp.max(), imp.min()
        imp -= min_imp
        imp /= float(max(max_imp - min_imp, 1e-6))
        
    return importance


def padded_op(p1, p2, op='-'):
    """
    Return the op between p1 and p2. Result size is size(p2).
    If p1 and p2 sizes are different, simply compute the difference 
    by cutting away additional values and zero-pad result to obtain back the p2 dimension.

    :param op: '-', '+', '*', '/' for difference, sum, multiplication, division
    """

    if p1.size() == p2.size():
        if op == '-':
            result = p1 - p2
        elif op == '+':
            result = p1 + p2
        if op == '*':
            result = p1 * p2
        if op == '/':
            result = p1 / p2
        return result

    assert len(p1.size()) == len(p2.size()) < 3, "CNN not supported for padded_op"

    min_size = torch.Size([
        min(a, b)
        for a,b in zip(p1.size(), p2.size())
    ])
    if len(p1.size()) == 2:
        resizedp1 = p1[:min_size[0], :min_size[1]]
        resizedp2 = p2[:min_size[0], :min_size[1]]
    else:
        resizedp1 = p1[:min_size[0]]
        resizedp2 = p2[:min_size[0]]

    if op == '-':
        result = resizedp1 - resizedp2
    elif op == '+':
        result = resizedp1 + resizedp2
    if op == '*':
        result = resizedp1 * resizedp2
    if op == '/':
        result = resizedp1 / resizedp2

    padded_result = torch.zeros(p2.size(), device=p2.device)
    if len(p1.size()) == 2:
        padded_result[:result.size(0), :result.size(1)] = result
    else:
        padded_result[:result.size(0)] = result

    return padded_result


def zerolike_params_dict(model, to_cpu=False):
    if to_cpu:
        return [ ( k, torch.zeros_like(p).to('cpu') ) for k,p in model.named_parameters() ]
    else:
        return [ ( k, torch.zeros_like(p).to(p.device) ) for k,p in model.named_parameters() ]


def copy_params_dict(model, copy_grad=False, to_cpu=False):
    if copy_grad:
        if to_cpu:
            return [ ( k, p.grad.cpu().data.clone() ) for k,p in model.named_parameters() ]
        else:
            return [ ( k, p.grad.data.clone() ) for k,p in model.named_parameters() ]
    else:
        if to_cpu:
            return [ ( k, p.cpu().data.clone() ) for k,p in model.named_parameters() ]
        else:
            return [ ( k, p.data.clone() ) for k,p in model.named_parameters() ]


def distillation_loss(out, prev_out, temperature):
    log_p = torch.log_softmax(out / temperature, dim=1)
    q = torch.softmax(prev_out / temperature, dim=1)
    res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
    return res
