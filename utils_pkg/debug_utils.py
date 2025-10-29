import torch
import torch.nn as nn
from typing import Dict, List
from collections import defaultdict


class AttentionDebugHook:
    """Hook để debug input/output của MultiheadAttention (tiết kiệm RAM).

    Mặc định KHÔNG lưu tensor, chỉ lưu thống kê (shape/mean/std). Có thể giới hạn
    số bản ghi bằng `max_records`. Nếu thật sự cần tensor, bật `keep_tensors=True`
    (khi đó sẽ ép về CPU+FP16 và chỉ giữ tối đa `max_records` bản ghi).
    """
    def __init__(self,
                 module_name: str = "",
                 keep_tensors: bool = False,
                 track_weights: bool = False,
                 to_cpu: bool = True,
                 tensor_dtype: torch.dtype = torch.float16,
                 max_records: int = 1):
        self.module_name = module_name
        self.keep_tensors = keep_tensors
        self.track_weights = track_weights
        self.to_cpu = to_cpu
        self.tensor_dtype = tensor_dtype
        self.max_records = max_records
        self.stats = {
            'inputs': [],
            'outputs': [],
            'weights': [],  # attention weights nếu có
        }
        self.enabled = True

    def register(self, module: nn.MultiheadAttention):
        """Đăng ký hook trên MultiheadAttention"""
        handle = module.register_forward_hook(self._forward_hook)
        return handle

    def _forward_hook(self, module, input_tuple, output):
        if not self.enabled:
            return
        Q, K, V = input_tuple[0], input_tuple[1], input_tuple[2]
        attn_out, attn_weights = output

        def _maybe_tensor(x):
            if not self.keep_tensors:
                return {'shape': tuple(x.shape), 'mean': x.mean().item(), 'std': x.std().item()}
            t = x.detach()
            if self.to_cpu:
                t = t.to('cpu')
            if self.tensor_dtype is not None:
                t = t.to(self.tensor_dtype)
            return t.contiguous()

        self.stats['inputs'].append({
            'Q': _maybe_tensor(Q),
            'K': _maybe_tensor(K),
            'V': _maybe_tensor(V),
        })
        self.stats['outputs'].append(_maybe_tensor(attn_out))
        if self.track_weights and (attn_weights is not None):
            self.stats['weights'].append(_maybe_tensor(attn_weights))

        # Giới hạn số bản ghi để tránh phình RAM
        for k in ['inputs', 'outputs', 'weights']:
            if len(self.stats[k]) > self.max_records:
                self.stats[k] = self.stats[k][-self.max_records:]

    def get_stats(self):
        """Lấy thống kê và reset"""
        stats = self.stats.copy()
        self.stats = {'inputs': [], 'outputs': [], 'weights': []}
        return stats

    def print_stats(self, step: int = 0):
        """In thống kê input/output"""
        if len(self.stats['inputs']) == 0:
            return
        last_in = self.stats['inputs'][-1]
        last_out = self.stats['outputs'][-1]
        print(f"[{self.module_name}] Step {step}:")
        def _fmt(x):
            if isinstance(x, dict):
                return f"shape={x['shape']}, mean={x['mean']:.4f}, std={x['std']:.4f}"
            return f"shape={tuple(x.shape)}, mean={x.mean().item():.4f}, std={x.std().item():.4f}"
        print(f"  Q: {_fmt(last_in['Q'])}")
        print(f"  K: {_fmt(last_in['K'])}")
        print(f"  V: {_fmt(last_in['V'])}")
        print(f"  Out: {_fmt(last_out if not isinstance(last_out, dict) else last_out)}")
        if len(self.stats['weights']) > 0:
            w = self.stats['weights'][-1]
            print(f"  AttnWeights: {_fmt(w if not isinstance(w, dict) else w)}")


class GradientDebugHook:
    """Hook để debug gradient (vanishing/exploding)"""
    def __init__(self, module_name: str = ""):
        self.module_name = module_name
        self.grad_stats = defaultdict(list)
        self.enabled = True

    def register(self, module: nn.Module):
        """Đăng ký hook trên tất cả parameters của module"""
        handles = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._grad_hook(name))
                handles.append(handle)
        return handles

    def _grad_hook(self, param_name: str):
        """Tạo hook cho một parameter"""
        def hook(grad):
            if grad is None or not self.enabled:
                return
            norm = grad.norm().item()
            mean = grad.mean().item()
            std = grad.std().item()
            max_abs = grad.abs().max().item()
            self.grad_stats[f"{self.module_name}.{param_name}"].append({
                'norm': norm,
                'mean': mean,
                'std': std,
                'max_abs': max_abs,
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
            })
        return hook

    def get_stats(self):
        """Lấy thống kê gradient và reset"""
        stats = dict(self.grad_stats)
        self.grad_stats = defaultdict(list)
        return stats

    def print_stats(self, step: int = 0, threshold_norm: float = 1e-6):
        """In thống kê gradient, cảnh báo vanishing nếu norm quá nhỏ"""
        if len(self.grad_stats) == 0:
            return
        print(f"[{self.module_name}] Gradient Stats (Step {step}):")
        for param_path, values in self.grad_stats.items():
            if len(values) == 0:
                continue
            last = values[-1]
            norm = last['norm']
            status = "⚠️ VANISHING" if norm < threshold_norm else "OK"
            print(f"  {param_path}: norm={norm:.6e}, mean={last['mean']:.6e}, std={last['std']:.6e}, max={last['max_abs']:.6e} {status}")
            if last['has_nan']:
                print(f"    ⚠️ Contains NaN!")
            if last['has_inf']:
                print(f"    ⚠️ Contains Inf!")


def register_debug_hooks(model: nn.Module, 
                         attn_modules: Dict[str, nn.Module] = None,
                         enable_attn_debug: bool = True,
                         enable_grad_debug: bool = True):
    """
    Đăng ký debug hooks cho model.
    
    Args:
        model: Main model
        attn_modules: Dict {name: module} của các attention modules cần debug
        enable_attn_debug: Bật debug attention input/output
        enable_grad_debug: Bật debug gradient
    
    Returns:
        Tuple (attn_hooks, grad_hooks, all_handles)
    """
    attn_hooks = {}
    grad_hooks = {}
    all_handles = []

    if enable_attn_debug and attn_modules:
        for name, module in attn_modules.items():
            hook = AttentionDebugHook(module_name=name)
            if isinstance(module, nn.MultiheadAttention):
                h = hook.register(module)
                all_handles.append(h)
                attn_hooks[name] = hook

    if enable_grad_debug:
        # Debug gradient cho toàn bộ model
        grad_hook = GradientDebugHook(module_name="model")
        handles = grad_hook.register(model)
        all_handles.extend(handles)
        grad_hooks["model"] = grad_hook

    return attn_hooks, grad_hooks, all_handles


def print_all_stats(attn_hooks: Dict, grad_hooks: Dict, step: int = 0):
    """In tất cả stats từ các hooks"""
    for name, hook in attn_hooks.items():
        hook.print_stats(step=step)
    for name, hook in grad_hooks.items():
        hook.print_stats(step=step)

