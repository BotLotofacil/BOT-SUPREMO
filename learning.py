# learning.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple
from datetime import datetime

STATE_PATH_DEFAULT = os.environ.get("LEARN_STATE_PATH", "learn_state.json")
N_UNIVERSO = 25
N_DEZENAS = 15


def _clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


def _hits(aposta: List[int], oficial: List[int]) -> int:
    return len(set(aposta) & set(oficial))


def _hits_media_topk(placares: List[int], k: int = 5) -> float:
    if not placares:
        return 0.0
    k = max(1, min(k, len(placares)))
    return sum(sorted(placares, reverse=True)[:k]) / float(k)


@dataclass
class LearnConfig:
    alpha_min: float = 0.30
    alpha_max: float = 0.50
    alpha_init: float = 0.36
    janela: int = 60
    # thresholds
    media_ok: float = 11.0     # média alvo
    topk_ok: float = 12.0      # top-k médio alvo
    # deltas
    bias_hit_delta: float = 0.35
    bias_miss_delta: float = -0.20
    alpha_up: float = 0.01
    alpha_down: float = -0.005


@dataclass
class LearnState:
    alpha: float
    janela: int
    bias_num: Dict[int, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def inicial(cfg: LearnConfig) -> "LearnState":
        bias = {i: 0.0 for i in range(1, N_UNIVERSO + 1)}
        return LearnState(
            alpha=cfg.alpha_init,
            janela=cfg.janela,
            bias_num=bias,
            history=[],
        )


class LearningCore:
    """
    Núcleo de aprendizado robusto:
    - Mantém alpha, janela e bias_num
    - Aplica reforço apenas quando lote é realmente bom
    - Penaliza levemente lotes ruins
    """

    def __init__(self, path: str = STATE_PATH_DEFAULT, cfg: LearnConfig | None = None) -> None:
        self.path = path
        self.cfg = cfg or LearnConfig()
        self.state: LearnState = self._carregar()

    # ------------------------ Persistência ------------------------

    def _carregar(self) -> LearnState:
        if not os.path.exists(self.path):
            return LearnState.inicial(self.cfg)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return LearnState.inicial(self.cfg)

        alpha = float(raw.get("alpha", self.cfg.alpha_init))
        janela = int(raw.get("janela", self.cfg.janela))
        bias_raw = raw.get("bias_num", {}) or {}
        bias = {int(k): float(v) for k, v in bias_raw.items() if str(k).isdigit()}

        hist = raw.get("history", []) or []
        # garante que não exploda em tamanho
        hist = hist[-self.cfg.janela :]

        st = LearnState(alpha=alpha, janela=janela, bias_num=bias, history=hist)
        # completa dezena faltante
        for i in range(1, N_UNIVERSO + 1):
            st.bias_num.setdefault(i, 0.0)
        return st

    def _salvar(self) -> None:
        raw = asdict(self.state)
        raw["bias_num"] = {int(k): float(v) for k, v in self.state.bias_num.items()}
        raw["history"] = list(self.state.history)[-self.cfg.janela :]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    # ------------------------ API pública ------------------------

    def get_bias_num(self) -> Dict[int, float]:
        return dict(self.state.bias_num)

    def get_alpha(self) -> float:
        return float(self.state.alpha)

    def registrar_lote_gerado(
        self,
        oficial_base: List[int],
        apostas: List[List[int]],
        tag: str = "geracao",
    ) -> None:
        """
        Apenas registra que um lote foi gerado (para histórico e auditoria).
        Não move bias aqui; o ajuste real acontece em aprender_com_lote.
        """
        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tag": tag,
            "oficial_base": " ".join(f"{n:02d}" for n in oficial_base),
            "qtd": len(apostas),
        }
        self.state.history.append(rec)
        self.state.history = self.state.history[-self.cfg.janela :]
        self._salvar()

    def aprender_com_lote(
        self,
        oficial: List[int],
        apostas: List[List[int]],
        tag: str = "aprender",
    ) -> Dict[str, Any]:
        """
        Avalia o lote com o resultado oficial e atualiza:
        - bias_num
        - alpha

        Retorna um resumo com métricas para exibir no bot.
        """
        oficial = sorted(set(int(x) for x in oficial if 1 <= int(x) <= N_UNIVERSO))
        if len(oficial) != N_DEZENAS:
            raise ValueError("Resultado oficial inválido para aprendizado.")

        placares = [_hits(a, oficial) for a in apostas]
        media = sum(placares) / len(placares) if placares else 0.0
        topk = _hits_media_topk(placares, k=min(5, len(placares))) if placares else 0.0
        melhor = max(placares) if placares else 0

        # Gating: só reforça forte quando o lote é bom
        lote_bom = (media >= self.cfg.media_ok) or (topk >= self.cfg.topk_ok)

        # Atualiza bias_num
        if lote_bom:
            # reforça dezenas presentes nos melhores bilhetes E no oficial
            sorted_idx = sorted(range(len(placares)), key=lambda i: placares[i], reverse=True)
            k_top = max(1, min(5, len(sorted_idx)))
            idx_top = sorted_idx[:k_top]
            count_num: Dict[int, int] = {i: 0 for i in range(1, N_UNIVERSO + 1)}
            for i in idx_top:
                for d in apostas[i]:
                    if d in oficial:
                        count_num[d] += 1

            for d in range(1, N_UNIVERSO + 1):
                if d in oficial and count_num.get(d, 0) > 0:
                    self.state.bias_num[d] = _clamp(
                        self.state.bias_num.get(d, 0.0) + self.cfg.bias_hit_delta,
                        -5.0,
                        5.0,
                    )
                else:
                    # leve penalização para quem não ajudou
                    self.state.bias_num[d] = _clamp(
                        self.state.bias_num.get(d, 0.0) + self.cfg.bias_miss_delta * 0.2,
                        -5.0,
                        5.0,
                    )
        else:
            # lote fraco: corrige levemente todos na direção do zero
            for d in range(1, N_UNIVERSO + 1):
                v = self.state.bias_num.get(d, 0.0)
                if abs(v) < 0.05:
                    self.state.bias_num[d] = 0.0
                elif v > 0:
                    self.state.bias_num[d] = v + self.cfg.bias_miss_delta * 0.5
                else:
                    self.state.bias_num[d] = v - self.cfg.bias_miss_delta * 0.5
                self.state.bias_num[d] = _clamp(self.state.bias_num[d], -5.0, 5.0)

        # Atualiza alpha
        if lote_bom:
            self.state.alpha = _clamp(
                self.state.alpha + self.cfg.alpha_up,
                self.cfg.alpha_min,
                self.cfg.alpha_max,
            )
        else:
            self.state.alpha = _clamp(
                self.state.alpha + self.cfg.alpha_down,
                self.cfg.alpha_min,
                self.cfg.alpha_max,
            )

        # registra no histórico
        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tag": tag,
            "media": media,
            "topk": topk,
            "melhor": melhor,
            "alpha": self.state.alpha,
        }
        self.state.history.append(rec)
        self.state.history = self.state.history[-self.cfg.janela :]

        self._salvar()

        return {
            "media": media,
            "topk": topk,
            "melhor": melhor,
            "alpha": self.state.alpha,
            "lote_bom": lote_bom,
            "placares": placares,
        }
