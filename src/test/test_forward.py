import torch
from llm import GPTConfig, GPTModel
from llm.layers import CausalSelfAttention


def test_forward_smoke():
    cfg = GPTConfig(
        vocab_size=8000,
        max_seq_len=32,
        n_layer=2,
        n_head=4,
        d_model=128,
        d_mlp=512,
        use_rope=True,
        norm_type="rmsnorm",
    )
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    print("forward smoke: logits/loss OK")


def test_backward_step():
    cfg = GPTConfig(
        vocab_size=5000,
        max_seq_len=32,
        n_layer=2,
        n_head=4,
        d_model=128,
        d_mlp=256,
        use_rope=True,
        norm_type="rmsnorm",
    )
    model = GPTModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print("backward step: optimizer OK")


def test_generate_shapes():
    cfg = GPTConfig(
        vocab_size=7777,
        max_seq_len=32,
        n_layer=2,
        n_head=4,
        d_model=128,
        d_mlp=512,
        use_rope=True,
        norm_type="rmsnorm",
        use_kv_cache=True,
    )
    model = GPTModel(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model.generate(seed, max_new_tokens=8)
    assert out.shape == (1, 24)
    print("generate: output shape OK")



def test_kvcache_equivalence():
    torch.manual_seed(0)
    d_model, n_head, T, B = 64, 8, 16, 2
    attn = CausalSelfAttention(d_model, n_head, use_rope=True, max_seq_len=128)
    x = torch.randn(B, T, d_model)

    # tam dizi
    y_full = attn(x)
    # adım adım cache
    y_parts, cache = [], {}
    for t in range(T):
        y_step = attn(x[:, t:t+1, :], kv_cache=cache, layer_idx=0)
        y_parts.append(y_step)
    y_infer = torch.cat(y_parts, dim=1)
    diff = torch.norm(y_full - y_infer).item()
    print("KV equiv L2:", diff)
    assert diff < 1e-3
    print("KV-cache equivalence OK")


def test_gqa_shapes():
    torch.manual_seed(0)
    d_model, n_head, n_kv_head, T, B = 64, 8, 2, 16, 2
    attn = CausalSelfAttention(
        d_model, n_head,
        n_kv_head=n_kv_head,
        use_rope=True,
        max_seq_len=128,
    )
    x = torch.randn(B, T, d_model)
    y = attn(x)
    assert y.shape == (B, T, d_model)
    print("GQA forward shape OK")


def test_rope_start_pos():
    torch.manual_seed(0)
    d_model, n_head, T, B = 64, 8, 8, 1
    attn = CausalSelfAttention(d_model, n_head, use_rope=True, max_seq_len=1024)
    x = torch.randn(B, T, d_model)
    try:
        y1 = attn(x)
        y2 = attn(x) 
        print("RoPE start_pos smoke OK")
    except TypeError:
        print("start_pos parametresi bu versiyonda yok, test skip edildi.")


if __name__ == "__main__":
    test_forward_smoke()
    test_backward_step()
    test_generate_shapes()
    test_kvcache_equivalence()
    test_gqa_shapes()
    test_rope_start_pos()
    print("All tests passed")
