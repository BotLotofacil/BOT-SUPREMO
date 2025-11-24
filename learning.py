from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
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
    # Alpha com movimento suave
    alpha_min: float = 0.30
    alpha_max: float = 0.50
    alpha_init: float = 0.36

    # Memória REAL (quantos concursos recentes entram no cálculo)
    janela: int = 50

    # referencial inicial (usado só no comecinho, até ter histórico)
    media_ref: float = 9.8
    topk_ref: float = 11.0

    # Bias em faixa suave [-3 .. +3]
    bias_min: float = -3.0
    bias_max: float = 3.0
    bias_hit_delta: float = 0.20   # reforço quando lote é realmente bom
    bias_miss_delta: float = -0.10 # penalização leve

    # Movimentos de alpha bem suaves
    alpha_up: float = 0.005
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
    Núcleo de aprendizado REAL:
    - Mantém alpha, janela e bias_num
    - Usa memória de até 50 concursos (history)
    - Gating baseado em desempenho relativo (média/top-k vs histórico)
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
        bias: Dict[int, float] = {}
        for k, v in bias_raw.items():
            if str(k).isdigit():
                idx = int(k)
                if 1 <= idx <= N_UNIVERSO:
                    bias[idx] = _clamp(float(v), self.cfg.bias_min, self.cfg.bias_max)

        # completa quaisquer dezenas que não apareceram
        for i in range(1, N_UNIVERSO + 1):
            bias.setdefault(i, 0.0)

        hist = raw.get("history", []) or []
        hist = hist[-self.cfg.janela :]

        return LearnState(alpha=alpha, janela=janela, bias_num=bias, history=hist)

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
        O ajuste real acontece em aprender_com_lote.
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

    def _estatisticas_historicas(self) -> tuple[float | None, float | None]:
        """Retorna (media_hist, topk_hist) usando apenas entradas com métricas."""
        medias = []
        topks = []
        for rec in self.state.history:
            if isinstance(rec, dict) and "media" in rec and "topk" in rec:
                try:
                    medias.append(float(rec["media"]))
                    topks.append(float(rec["topk"]))
                except Exception:
                    continue
        media_hist = sum(medias) / len(medias) if medias else None
        topk_hist = sum(topks) / len(topks) if topks else None
        return media_hist, topk_hist

    def aprender_com_lote(
        self,
        oficial: List[int],
        apostas: List[List[int]],
        tag: str = "aprender",
    ) -> Dict[str, Any]:
        """
        Avalia o lote com o resultado oficial e atualiza:
        - bias_num (dezena a dezena)
        - alpha (força de repetição)

        Usa memória de até 50 concursos para decidir se o lote é:
        - "forte" (acima do histórico) → reforça
        - "fraco" (abaixo do histórico) → corrige
        - "neutro" → ajustes mínimos
        """
        oficial = sorted(set(int(x) for x in oficial if 1 <= int(x) <= N_UNIVERSO))
        if len(oficial) != N_DEZENAS:
            raise ValueError("Resultado oficial inválido para aprendizado.")

        placares = [_hits(a, oficial) for a in apostas]
        media = sum(placares) / len(placares) if placares else 0.0
        topk = _hits_media_topk(placares, k=min(5, len(placares))) if placares else 0.0
        melhor = max(placares) if placares else 0

        # Estatísticas históricas (memória REAL)
        media_hist, topk_hist = self._estatisticas_historicas()

        lote_bom = False
        lote_ruim = False

        if media_hist is not None and topk_hist is not None:
            # Comparação relativa: quanto este lote ficou acima/abaixo da média histórica
            delta_m = media - media_hist
            delta_t = topk - topk_hist

            # Gating suave
            if delta_m >= 0.4 or delta_t >= 0.6:
                lote_bom = True
            elif delta_m <= -0.4 and delta_t <= -0.6:
                lote_ruim = True
        else:
            # Ainda não temos histórico suficiente → usa thresholds de referência
            if media >= self.cfg.media_ref or topk >= self.cfg.topk_ref:
                lote_bom = True
            elif media <= (self.cfg.media_ref - 0.8) and topk <= (self.cfg.topk_ref - 1.0):
                lote_ruim = True

        # ------------------------ Atualiza bias_num ------------------------

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
                    # reforço suave positivo
                    novo = self.state.bias_num.get(d, 0.0) + self.cfg.bias_hit_delta
                    self.state.bias_num[d] = _clamp(novo, self.cfg.bias_min, self.cfg.bias_max)
                else:
                    # penalização leve para quem não ajudou
                    v = self.state.bias_num.get(d, 0.0)
                    v = v + self.cfg.bias_miss_delta * 0.3
                    self.state.bias_num[d] = _clamp(v, self.cfg.bias_min, self.cfg.bias_max)

        elif lote_ruim:
            # lote ruim → puxa tudo na direção do zero de forma mais firme
            for d in range(1, N_UNIVERSO + 1):
                v = self.state.bias_num.get(d, 0.0)
                v *= 0.85  # contrai
                if abs(v) < 0.03:
                    v = 0.0
                self.state.bias_num[d] = _clamp(v, self.cfg.bias_min, self.cfg.bias_max)
        else:
            # lote neutro → pequeno "relaxamento" em direção ao zero
            for d in range(1, N_UNIVERSO + 1):
                v = self.state.bias_num.get(d, 0.0)
                v *= 0.95
                if abs(v) < 0.02:
                    v = 0.0
                self.state.bias_num[d] = _clamp(v, self.cfg.bias_min, self.cfg.bias_max)

        # ------------------------ Atualiza alpha (suave) ------------------------

        if lote_bom:
            self.state.alpha = _clamp(
                self.state.alpha + self.cfg.alpha_up,
                self.cfg.alpha_min,
                self.cfg.alpha_max,
            )
        elif lote_ruim:
            self.state.alpha = _clamp(
                self.state.alpha + self.cfg.alpha_down,
                self.cfg.alpha_min,
                self.cfg.alpha_max,
            )
        else:
            # puxa levemente alpha de volta para o alpha_init (estabilidade)
            alvo = self.cfg.alpha_init
            self.state.alpha = _clamp(
                self.state.alpha + (alvo - self.state.alpha) * 0.05,
                self.cfg.alpha_min,
                self.cfg.alpha_max,
            )

        # ------------------------ Registra no histórico ------------------------

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
