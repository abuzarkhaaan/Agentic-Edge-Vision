from __future__ import annotations
import hashlib, json, os, secrets, struct, time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _require_crypto():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return True
    except ImportError:
        return False


def encrypt_payload(key_bytes: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = secrets.token_bytes(12)
    ct = AESGCM(key_bytes).encrypt(nonce, plaintext, None)
    return nonce, ct


def decrypt_payload(key_bytes: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key_bytes).decrypt(nonce, ciphertext, None)


def sign_message(private_key, message: bytes) -> bytes:
    return private_key.sign(message)


def verify_signature(public_key, signature: bytes, message: bytes) -> bool:
    try:
        public_key.verify(signature, message)
        return True
    except Exception:
        return False


def chain_hash(prev_v: bytes, payload: bytes) -> bytes:
    ph = hashlib.sha256(payload).digest()
    return hashlib.sha256(prev_v + ph).digest()


@dataclass
class AuditRecord:
    timestamp: float
    session_id: str
    seq_counter: int
    pred_class: int
    uncertainty_q: int
    action: str
    payload_digest: bytes
    chain_value: bytes
    signature: bytes
    gcm_tag: bytes


class SecureFeatureSender:
    PAYLOAD_BYTES = 800

    def __init__(self, aes_key: bytes | None = None):
        if not _require_crypto():
            raise ImportError("pip install cryptography")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.aes_key = aes_key or secrets.token_bytes(32)
        self._priv_key = Ed25519PrivateKey.generate()
        self.pub_key = self._priv_key.public_key()
        self._chain_v = secrets.token_bytes(32)
        self._seq = 0
        self._session_id = secrets.token_hex(6)
        self.log: list[AuditRecord] = []

    def send(self, embedding: np.ndarray, pred_class: int,
             uncertainty: float, action: str = "FeatureSend") -> bytes:
        self._seq += 1
        emb_bytes = embedding.astype(np.float32).tobytes()
        u_q = int(np.clip(uncertainty * 255, 0, 255))
        ts = time.time()

        meta = struct.pack(">dBBH", ts, pred_class, u_q, self._seq % 65536)
        meta += b"\x00" * 4

        plaintext = emb_bytes + meta
        nonce, ciphertext = encrypt_payload(self.aes_key, plaintext)

        self._chain_v = chain_hash(self._chain_v, ciphertext)
        sig_msg = ciphertext + self._chain_v
        signature = sign_message(self._priv_key, sig_msg)

        gcm_tag = ciphertext[-16:]
        record = AuditRecord(
            timestamp=ts, session_id=self._session_id,
            seq_counter=self._seq, pred_class=pred_class,
            uncertainty_q=u_q, action=action,
            payload_digest=hashlib.sha256(ciphertext).digest(),
            chain_value=self._chain_v, signature=signature,
            gcm_tag=gcm_tag,
        )
        self.log.append(record)

        wire = nonce + signature + self._chain_v + ciphertext
        return wire

    def verify_chain(self) -> bool:
        v = hashlib.sha256(b"genesis").digest()
        for rec in self.log:
            expected = chain_hash(v, rec.payload_digest)
            if expected != rec.chain_value:
                return False
            v = rec.chain_value
        return True


class InversionDecoder(nn.Module):
    def __init__(self, d: int = 192):
        super().__init__()
        self.fc = nn.Linear(d, 512 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 64,   3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 2, 2)
        return self.decoder(x)


def train_inversion_decoder(embeddings: torch.Tensor, thumbnails: torch.Tensor,
                             epochs: int = 60, lr: float = 1e-3,
                             device: str = "cpu") -> InversionDecoder:
    model = InversionDecoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(embeddings, thumbnails)
    ldr = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    for ep in range(epochs):
        for z, t in ldr:
            z, t = z.to(device), t.to(device)
            opt.zero_grad()
            loss_fn(model(z), t).backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  Inversion epoch {ep + 1}/{epochs}")
    return model


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    return float(20 * np.log10(1.0 / (np.sqrt(mse) + 1e-9)))


def ssim_approx(pred: np.ndarray, target: np.ndarray) -> float:
    mu_p, mu_t = pred.mean(), target.mean()
    sigma_p = pred.std()
    sigma_t = target.std()
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return float(
        ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) /
        ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p ** 2 + sigma_t ** 2 + C2))
    )


def membership_auc(train_emb: np.ndarray, test_emb: np.ndarray,
                   threshold: float = 0.5) -> float:
    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([train_emb.max(axis=1), test_emb.max(axis=1)])
    labels = np.concatenate([np.ones(len(train_emb)), np.zeros(len(test_emb))])
    return float(roc_auc_score(labels, scores))
