import torch

def foa_to_iv(foa_wave, sr=16_000, n_fft=400, hop=100, eps=1e-6):
    B, C, T = foa_wave.shape
    win = torch.hann_window(n_fft, device=foa_wave.device)

    spec = torch.stft(
        foa_wave.view(-1, T),
        n_fft=n_fft, hop_length=hop,
        window=win, center=True,
        return_complex=True,
    ).view(B, 4, n_fft//2+1, -1)      # (B,4,201,1601)

    W, Y, Z, X = spec[:,0], spec[:,1], spec[:,2], spec[:,3]
    conjW = W.conj()

    I_act = torch.stack([(conjW*Y).real,
                         (conjW*Z).real,
                         (conjW*X).real], dim=1)
    I_rea = torch.stack([(conjW*Y).imag,
                         (conjW*Z).imag,
                         (conjW*X).imag], dim=1)

    # --- unit-norm：ゼロベクトルはそのまま ---
    norm = torch.linalg.norm(I_act, dim=1, keepdim=True)          # (B,1,F,T)
    I_act = torch.where(norm > eps, I_act / norm, I_act)          # (B,3,F,T)
    I_rea = torch.where(norm > eps, I_rea / norm, I_rea)

    return I_act.float(), I_rea.float()
