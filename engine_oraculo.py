# engine_oraculo.py
# Motor de geração de apostas para Lotofácil
# Foco: shape Mestre (paridade 7–8, seq<=3), anti-overlap, repetição alta.

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

N_UNIVERSO = 25
N_DEZENAS = 15
OVERLAP_MAX_DEFAULT = 11


def max_seq(dezenas: List[int]) -> int:
    """Retorna o tamanho da maior sequência consecutiva em uma aposta."""
    if not dezenas:
        return 0
    dezenas = sorted(dezenas)
    maior = 1
    atual = 1
    for i in range(1, len(dezenas)):
        if dezenas[i] == dezenas[i - 1] + 1:
            atual += 1
            if atual > maior:
                maior = atual
        else:
            atual = 1
    return maior


def paridade(dezenas: List[int]) -> Tuple[int, int]:
    """Retorna (pares, ímpares)."""
    pares = sum(1 for d in dezenas if d % 2 == 0)
    impares = len(dezenas) - pares
    return pares, impares


def shape_ok_mestre(dezenas: List[int]) -> bool:
    """Shape Mestre: 7–8 pares, SeqMax ≤ 3."""
    pares, _ = paridade(dezenas)
    seq = max_seq(dezenas)
    return 7 <= pares <= 8 and seq <= 3


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


@dataclass
class EngineConfig:
    overlap_max: int = OVERLAP_MAX_DEFAULT
    target_qtd: int = 15
    # peso base da repetição R (quantidade de dezenas repetidas do último resultado)
    base_R_plan: Tuple[int, ...] = (10, 9, 9, 10, 10, 9, 10, 8, 11, 11, 10, 9, 11, 8, 10)


class OraculoEngine:
    """
    Motor de geração:
    - Usa o último resultado (L) + complemento (C)
    - Usa plano de repetição R (9R–10R alvo, com 8R e 11R para variação)
    - Aplica vieses de dezenas (bias_num)
    - Enforça shape Mestre e anti-overlap
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        bias_num: Dict[int, float] | None = None,
        alpha: float = 0.36,
    ) -> None:
        self.config = config or EngineConfig()
        # bias_num: 1..25 -> peso (0.0 ~ 1.0+)
        self.bias_num: Dict[int, float] = {i: 0.0 for i in range(1, N_UNIVERSO + 1)}
        if bias_num:
            for k, v in bias_num.items():
                if 1 <= int(k) <= N_UNIVERSO:
                    self.bias_num[int(k)] = float(v)
        self.alpha = float(alpha)

    # ------------------------- Helpers internos -------------------------

    def _weights_for_group(self, dezenas: List[int]) -> List[float]:
        """Transforma bias_num em pesos positivos para um grupo de dezenas."""
        pesos = []
        for d in dezenas:
            b = self.bias_num.get(d, 0.0)
            # mapeia bias [-5, +5] para peso [0.5, 2.0] (exemplo)
            w = 1.0 + 0.15 * b
            if w < 0.1:
                w = 0.1
            pesos.append(w)
        soma = sum(pesos)
        if soma <= 0:
            return [1.0 for _ in pesos]
        return [w / soma for w in pesos]

    def _sample_weighted(
        self, rng: random.Random, dezenas: List[int], k: int
    ) -> List[int]:
        """Sorteia k dezenas distintas de 'dezenas' usando pesos vindos do bias."""
        if k <= 0 or not dezenas:
            return []
        dezenas = list(sorted(set(dezenas)))
        if k >= len(dezenas):
            return list(dezenas)
        pesos = self._weights_for_group(dezenas)
        escolhidos = []
        disponiveis = list(dezenas)
        w = list(pesos)
        for _ in range(k):
            # roleta viciada
            r = rng.random()
            acum = 0.0
            idx = 0
            for i, p in enumerate(w):
                acum += p
                if r <= acum:
                    idx = i
                    break
            escolhidos.append(disponiveis[idx])
            disponiveis.pop(idx)
            w.pop(idx)
            if not disponiveis:
                break
            # renormaliza
            s = sum(w)
            if s <= 0:
                w = [1.0 / len(w) for _ in w]
            else:
                w = [x / s for x in w]
        return sorted(escolhidos)

    def _ajustar_shape_local(self, rng: random.Random, aposta: List[int]) -> List[int]:
        """
        Tenta ajustar paridade 7–8 e SeqMax≤3 trocando algumas dezenas com o universo.
        É heurístico, mas rápido.
        """
        aposta = sorted(set(aposta))
        if len(aposta) != N_DEZENAS:
            return aposta
        pares, imp = paridade(aposta)
        universo = list(range(1, N_UNIVERSO + 1))
        fora = [n for n in universo if n not in aposta]

        # Primeiro tenta quebrar sequências longas
        for _ in range(4):
            if max_seq(aposta) <= 3:
                break
            # escolhe um número de uma sequência longa para retirar
            seq_atual_max = max_seq(aposta)
            if seq_atual_max <= 3:
                break
            # remove um número aleatório da aposta
            idx_remove = rng.randrange(len(aposta))
            rem = aposta[idx_remove]
            aposta.pop(idx_remove)
            # adiciona algum número de fora
            if fora:
                add = rng.choice(fora)
                fora.remove(add)
                aposta.append(add)
                aposta.sort()

        # Ajuste de paridade
        pares, imp = paridade(aposta)
        for _ in range(4):
            if 7 <= pares <= 8 and max_seq(aposta) <= 3:
                break
            # se tem pares demais, troca um par por um ímpar de fora, e vice-versa
            if pares > 8:
                candidatos = [d for d in aposta if d % 2 == 0]
                if not candidatos or not fora:
                    break
                rem = rng.choice(candidatos)
                aposta.remove(rem)
                fora.append(rem)
                imp_candidates = [x for x in fora if x % 2 == 1]
                if not imp_candidates:
                    # volta
                    aposta.append(rem)
                    fora.remove(rem)
                    break
                add = rng.choice(imp_candidates)
                fora.remove(add)
                aposta.append(add)
                aposta.sort()
            elif pares < 7:
                candidatos = [d for d in aposta if d % 2 == 1]
                if not candidatos or not fora:
                    break
                rem = rng.choice(candidatos)
                aposta.remove(rem)
                fora.append(rem)
                par_candidates = [x for x in fora if x % 2 == 0]
                if not par_candidates:
                    aposta.append(rem)
                    fora.remove(rem)
                    break
                add = rng.choice(par_candidates)
                fora.remove(add)
                aposta.append(add)
                aposta.sort()

            pares, imp = paridade(aposta)

        return sorted(set(aposta))

    # ------------------------- Geração principal -------------------------

    def gerar_lote(
        self,
        ultimo_resultado: List[int],
        qtd: int | None = None,
        seed: int | None = None,
    ) -> List[List[int]]:
        """
        Gera um lote de apostas:
        - Usa plano de repetição R, com viés leve por alpha/bias se desejar
        - Enforça shape Mestre e anti-overlap
        """
        if qtd is None:
            qtd = self.config.target_qtd

        # Normaliza último resultado
        L = sorted(set(int(x) for x in ultimo_resultado if 1 <= int(x) <= N_UNIVERSO))
        if len(L) != N_DEZENAS:
            raise ValueError("último resultado inválido: precisa de 15 dezenas 1–25.")

        universo = list(range(1, N_UNIVERSO + 1))
        C = [n for n in universo if n not in L]

        r = random.Random(seed)

        # Plano de repetição R (podemos “puxar” para 10R com alpha)
        base_R = list(self.config.base_R_plan)
        plano_R: List[int] = []
        for i in range(qtd):
            # leve ajuste: quanto maior alpha, maior viés para 10R/11R
            idx = i % len(base_R)
            val = base_R[idx]
            if self.alpha > 0.35:
                if val == 9 and r.random() < (self.alpha - 0.35):
                    val = 10
                if val == 10 and r.random() < (self.alpha - 0.35) / 2:
                    val = 11
            plano_R.append(val)

        apostas: List[List[int]] = []

        # Geração com controle de overlap
        for i in range(qtd * 4):  # limite de tentativas
            if len(apostas) >= qtd:
                break

            R_target = plano_R[len(apostas)]
            R_target = max(6, min(13, R_target))  # segurança

            # escolha ponderada com bias
            from_last = self._sample_weighted(r, L, R_target)
            from_comp = self._sample_weighted(r, C, N_DEZENAS - len(from_last))
            aposta = sorted(set(from_last + from_comp))

            # segurança de tamanho
            if len(aposta) < N_DEZENAS:
                # completa randomicamente do universo
                faltam = N_DEZENAS - len(aposta)
                pool = [n for n in universo if n not in aposta]
                r.shuffle(pool)
                aposta.extend(pool[:faltam])
                aposta = sorted(set(aposta))

            aposta = self._ajustar_shape_local(r, aposta)

            # shape Mestre
            if not shape_ok_mestre(aposta):
                continue

            # anti-overlap
            violou = False
            for b in apostas:
                if len(set(aposta) & set(b)) > self.config.overlap_max:
                    violou = True
                    break
            if violou:
                continue

            # dedup
            if any(set(aposta) == set(b) for b in apostas):
                continue

            apostas.append(sorted(aposta))

        # Se, por algum motivo, veio menos que o alvo, apenas retorna o que conseguiu.
        return [sorted(a) for a in apostas]
