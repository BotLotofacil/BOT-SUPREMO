# engine_oraculo.py
# Motor de geração de apostas para Lotofácil
# Foco: shape Mestre (paridade 7–8, seq<=3), anti-overlap, repetição alta.
#
# ✅ Patch 2026-02-10: "Portfólio/otimizador do bloco"
# - Em vez de criar 10 apostas diretamente (campo estreito), o motor agora:
#   (1) gera um pool grande de candidatos determinísticos (centenas/milhares)
#   (2) seleciona o bloco de 10 como um PORTFÓLIO, com:
#       • penalidade por interseção (anti-colapso)
#       • recompensa por cobertura global do complemento (ausentes do último resultado)
#       • preserva o plano Mestre (9R–10R base + 1x 8R + 2x 11R por default do seu plano)
# - Continua 100% determinístico: mesma base/seed => mesmo lote.

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

N_UNIVERSO = 25
N_DEZENAS = 15
OVERLAP_MAX_DEFAULT = 11


def deterministic_seed(base_dezenas: List[int], version: str = "mestre_v3", salt: str = "lotofacil") -> int:
    """Gera um seed determinístico (mesma base => mesmo seed) para garantir reprodutibilidade."""
    base = ",".join(str(int(x)) for x in sorted(base_dezenas))
    payload = f"{salt}|{version}|{base}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:16], 16)


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
    """(pares, ímpares)."""
    p = sum(1 for x in dezenas if x % 2 == 0)
    return p, len(dezenas) - p


def shape_ok_mestre(dezenas: List[int]) -> bool:
    """Shape Mestre: 7–8 pares, SeqMax ≤ 3."""
    pares, _ = paridade(dezenas)
    seq = max_seq(dezenas)
    return 7 <= pares <= 8 and seq <= 3


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


@dataclass
class EngineConfig:
    # anti-overlap (hard) entre apostas selecionadas (interseção máxima)
    overlap_max: int = OVERLAP_MAX_DEFAULT

    # alvo de quantidade de apostas no lote (Preset Mestre => 10)
    target_qtd: int = 10

    # plano base de repetição R (quantidade de dezenas repetidas do último resultado)
    # Usado ciclicamente conforme o índice da aposta no lote.
    
    base_R_plan: Tuple[int, ...] = (
        # ciclo de 10 (Preset Mestre): 8R=1x, 11R=1x, resto 9R–10R
        10, 9, 9, 10, 10,
        9, 10, 9, 8, 11,
        # repete o ciclo para alimentar o pool (determinístico)
        10, 9, 9, 10, 10,
        9, 10, 9, 8, 11,
    )

    # -------------------------------
    # NOVO: Portfólio / Otimizador
    # -------------------------------
    use_portfolio_optimizer: bool = True

    # tamanho do pool de candidatos (determinístico) para selecionar o bloco final
    candidate_pool_size: int = 1200  # 300–2000 é um bom range

    # penalidade por sobreposição no score (além do hard overlap_max)
    overlap_soft_max: int = 10  # acima disso, penaliza forte (mesmo se <= overlap_max)

    # peso da penalidade por overlap (quanto maior, mais diversidade)
    overlap_penalty: float = 0.55

    # recompensa por cobrir dezenas do complemento (ausentes do último resultado)
    coverage_reward: float = 0.85

    # cobertura mínima: todas as dezenas do complemento devem aparecer ao menos 1x no bloco
    require_full_complement_coverage: bool = True

    # reforço: garante pelo menos N jogos "contrarian" (mais complemento / menor R)
    contrarian_games: int = 2


@dataclass(frozen=True)
class _Candidate:
    dezenas: Tuple[int, ...]  # ordenadas, tamanho 15
    R: int                   # repetição vs L
    score_base: float        # score intrínseco do candidato
    complement_hits: int     # quantas dezenas do complemento C
    role: str                # "core" | "balance" | "contrarian" | "aggressive"


class OraculoEngine:
    """
    Motor de geração (Lotofácil - Mestre):

    Antes: construía 10 apostas em sequência, aplicando anti-overlap.
    Problema: o lote ficava estreito => colapso (todas caem juntas).

    Agora (default):
    - Gera um pool grande de candidatos determinísticos
    - Escolhe o bloco de 10 como um PORTFÓLIO (greedy determinístico),
      maximizando score + cobertura e minimizando overlap.

    Regras invariantes:
    - paridade 7–8
    - seq<=3
    - anti-overlap hard (<=overlap_max)
    - determinismo: mesma base/seed => mesmo lote
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        bias_num: Optional[Dict[int, float]] = None,
        alpha: float = 0.36,
    ) -> None:
        self.config = config or EngineConfig()

        # bias_num: 1..25 -> peso
        self.bias_num: Dict[int, float] = {i: 0.0 for i in range(1, N_UNIVERSO + 1)}
        if bias_num:
            for k, v in bias_num.items():
                try:
                    idx = int(k)
                    if 1 <= idx <= N_UNIVERSO:
                        self.bias_num[idx] = float(v)
                except Exception:
                    continue
        self.alpha = float(alpha)

    # ------------------------- Helpers internos -------------------------

    def _weights_for_group(self, dezenas: List[int]) -> List[float]:
        """Transforma bias_num em pesos positivos para um grupo de dezenas."""
        pesos = []
        for d in dezenas:
            b = self.bias_num.get(d, 0.0)
            # bias em [-3, +3] → mapeia para algo em torno de [~0.55, ~1.45]
            w = 1.0 + 0.15 * b
            if w < 0.1:
                w = 0.1
            pesos.append(w)
        soma = sum(pesos)
        if soma <= 0:
            return [1.0 for _ in pesos]
        return [w / soma for w in pesos]

    def _sample_weighted(self, rng: random.Random, dezenas: List[int], k: int) -> List[int]:
        """Sorteia k dezenas distintas usando pesos vindos do bias (determinístico pelo rng)."""
        if k <= 0 or not dezenas:
            return []
        dezenas = list(sorted(set(dezenas)))
        if k >= len(dezenas):
            return list(dezenas)

        pesos = self._weights_for_group(dezenas)

        escolhidos: List[int] = []
        disponiveis = list(dezenas)
        w = list(pesos)

        for _ in range(k):
            r = rng.random()
            acum = 0.0
            idx = 0
            for i, p in enumerate(w):
                acum += p
                if r <= acum:
                    idx = i
                    break

            escolhidos.append(disponiveis[idx])

            # remove escolhido e renormaliza
            disponiveis.pop(idx)
            w.pop(idx)
            s = sum(w)
            if s > 0:
                w = [x / s for x in w]
            else:
                w = [1.0 / len(w) for _ in w] if w else []

        return escolhidos

    def _ajustar_shape_local(self, rng: random.Random, aposta: List[int]) -> List[int]:
        """Ajuste local para bater paridade/seq sem quebrar o determinismo."""
        aposta = sorted(set(int(x) for x in aposta if 1 <= int(x) <= N_UNIVERSO))
        if len(aposta) != N_DEZENAS:
            # completa/fixa tamanho
            universo = list(range(1, N_UNIVERSO + 1))
            pool = [n for n in universo if n not in aposta]
            rng.shuffle(pool)
            aposta = sorted((set(aposta) | set(pool[: (N_DEZENAS - len(aposta))])))

        universo = list(range(1, N_UNIVERSO + 1))

        # tenta poucas iterações para corrigir
        for _ in range(60):
            if shape_ok_mestre(aposta):
                return sorted(aposta)

            pares, _ = paridade(aposta)
            seq = max_seq(aposta)

            # quebra sequência >3 trocando 1 elemento do meio
            if seq > 3:
                a = sorted(aposta)
                # acha uma sequência grande
                for i in range(len(a) - 3):
                    if a[i + 3] == a[i] + 3:
                        # remove um do meio (i+1 ou i+2)
                        rem = a[i + 1] if rng.random() < 0.5 else a[i + 2]
                        pool = [n for n in universo if n not in aposta]
                        rng.shuffle(pool)
                        if pool:
                            aposta = sorted((set(aposta) - {rem}) | {pool[0]})
                        break
                continue

            # corrige paridade para 7–8 pares
            if pares < 7:
                # precisa adicionar par e remover ímpar
                odds = [x for x in aposta if x % 2 == 1]
                evens_pool = [x for x in universo if x % 2 == 0 and x not in aposta]
                rng.shuffle(odds); rng.shuffle(evens_pool)
                if odds and evens_pool:
                    aposta = sorted((set(aposta) - {odds[0]}) | {evens_pool[0]})
                continue

            if pares > 8:
                evens = [x for x in aposta if x % 2 == 0]
                odds_pool = [x for x in universo if x % 2 == 1 and x not in aposta]
                rng.shuffle(evens); rng.shuffle(odds_pool)
                if evens and odds_pool:
                    aposta = sorted((set(aposta) - {evens[0]}) | {odds_pool[0]})
                continue

            break

        return sorted(aposta)

    def _candidate_seed(self, base_seed: int, i: int, role: str) -> int:
        payload = f"{base_seed}|{i}|{role}".encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()
        return int(h[:16], 16)

    def _score_dezena(self, d: int) -> float:
        # score suave por dezena, derivado do bias aprendido
        b = float(self.bias_num.get(d, 0.0))
        return 1.0 + 0.10 * b

    def _build_candidate(
        self,
        L: List[int],
        C: List[int],
        base_seed: int,
        i: int,
        role: str,
        R_target: int,
    ) -> Optional[_Candidate]:
        rng = random.Random(self._candidate_seed(base_seed, i, role))

        # roles controlam quanta preferência ao complemento
        if role == "contrarian":
            # contrarian NÃO pode quebrar o Preset Mestre: mínimo 8R
            R_target = max(8, min(9, R_target))
        elif role == "aggressive":
            R_target = max(10, min(12, R_target))
        else:
            R_target = max(8, min(11, R_target))

        from_last = self._sample_weighted(rng, L, R_target)
        from_comp = self._sample_weighted(rng, C, N_DEZENAS - len(from_last))

        aposta = sorted(set(from_last + from_comp))

        if len(aposta) != N_DEZENAS:
            universo = list(range(1, N_UNIVERSO + 1))
            pool = [n for n in universo if n not in aposta]
            rng.shuffle(pool)
            aposta = sorted((set(aposta) | set(pool[: (N_DEZENAS - len(aposta))])))

        aposta = self._ajustar_shape_local(rng, aposta)
        if len(aposta) != N_DEZENAS:
            return None
        if not shape_ok_mestre(aposta):
            return None

        s = set(aposta)
        R = len(s & set(L))
        comp_hits = len(s & set(C))

        # score base (intrínseco do candidato):
        # - favorece bias positivo
        # - pequena recompensa por estar perto do R_target do role
        # - e por ter complemento (evitar campo estreito)
        bias_sum = sum(self._score_dezena(d) for d in aposta)
        r_bonus = 1.0 - (abs(R - R_target) / 6.0)  # 0..1
        comp_bonus = comp_hits / 10.0  # 0..1 (C tem 10 dezenas)

        score_base = (0.70 * bias_sum) + (4.0 * r_bonus) + (2.0 * comp_bonus)

        return _Candidate(tuple(sorted(aposta)), int(R), float(score_base), int(comp_hits), role)

    def _generate_candidate_pool(
        self,
        L: List[int],
        C: List[int],
        base_seed: int,
        qtd: int,
    ) -> List[_Candidate]:
        # mix de roles para gerar diversidade estrutural
        roles = []
        # core/balance dominam o pool
        roles.extend(["core"] * 6)
        roles.extend(["balance"] * 3)
        roles.extend(["contrarian"] * 2)
        roles.extend(["aggressive"] * 1)

        base_R = list(self.config.base_R_plan)

        pool: Dict[Tuple[int, ...], _Candidate] = {}

        # gera candidatos determinísticos; i varre e role escolhe
        for i in range(max(50, self.config.candidate_pool_size)):
            role = roles[i % len(roles)]
            # define R_target do candidato pela posição (mistura seu plano com role)
            R_target = base_R[i % len(base_R)]
            cand = self._build_candidate(L, C, base_seed, i, role, R_target)
            if cand is None:
                continue
            # mantém o melhor score caso duplicado
            if cand.dezenas not in pool or cand.score_base > pool[cand.dezenas].score_base:
                pool[cand.dezenas] = cand

            if len(pool) >= self.config.candidate_pool_size:
                break

        cands = list(pool.values())
        # ordena por score_base decrescente para acelerar greedy
        cands.sort(key=lambda c: c.score_base, reverse=True)
        return cands

    def _portfolio_greedy_select(
        self,
        candidates: List[_Candidate],
        L: List[int],
        C: List[int],
        qtd: int,
    ) -> List[List[int]]:
        if not candidates:
            raise RuntimeError("Pool de candidatos vazio. Não há como selecionar portfólio.")

        chosen: List[_Candidate] = []
        chosen_sets: List[set[int]] = []
        covered_C: set[int] = set()

        def overlap(a: set[int], b: set[int]) -> int:
            return len(a & b)

        # metas de composição do lote (mantém seu espírito Mestre, mas sem colapsar)
        #  - garante 2 contrarian e 1 agressivo, resto core/balance
        need_contrarian = max(0, int(self.config.contrarian_games))
        need_aggressive = 1  # pelo menos 1 jogo agressivo (11R-ish)
        # o resto se ajusta no greedy


        # quotas de repetição (Preset Mestre): 1x 8R, 1x 11R, resto 9R–10R
        target_R = {8: 1, 9: 4, 10: 4, 11: 1}
        r_count = {8: 0, 9: 0, 10: 0, 11: 0}

        def _r_penalty(r: int) -> float:
            # penaliza forte ultrapassar a quota para evitar "deslizar" para 6R/7R ou excesso de 8R/11R
            if r not in target_R:
                return 5.0
            if r_count[r] >= target_R[r]:
                return 2.5
            return 0.0

        # greedy determinístico: a cada passo escolhe o candidato que maximiza:
        # score_base + coverage_reward*(novas de C) - overlap_penalty*(overlaps)
        for step in range(qtd):
            best: Optional[_Candidate] = None
            best_score = -1e18

            for cand in candidates:
                if cand in chosen:
                    continue

                # enforce quotas cedo para garantir contrarian/aggressive
                if step < 6 and need_contrarian > 0 and cand.role != "contrarian":
                    # nas primeiras escolhas, ainda deixa passar, mas com penalidade
                    role_pen = 2.5
                else:
                    role_pen = 0.0

                if step < 5 and need_aggressive > 0 and cand.role != "aggressive":
                    role_pen += 1.8

                s = set(cand.dezenas)

                # hard overlap
                hard_violation = False
                for cs in chosen_sets:
                    if overlap(s, cs) > self.config.overlap_max:
                        hard_violation = True
                        break
                if hard_violation:
                    continue

                # guarda Preset Mestre: não permitir R < 8 (evita 6R/7R no bloco final)
                if cand.R < 8 or cand.R > 11:
                    continue
                r_pen = _r_penalty(cand.R)

                # penalidade soft por overlaps altos
                ov_pen = 0.0
                for cs in chosen_sets:
                    ov = overlap(s, cs)
                    if ov > self.config.overlap_soft_max:
                        ov_pen += (ov - self.config.overlap_soft_max) * 2.0
                    else:
                        ov_pen += ov * 0.25

                # recompensa de cobertura do complemento
                new_cov = len((s & set(C)) - covered_C)

                score = (
                    cand.score_base
                    + self.config.coverage_reward * (new_cov * 5.0)
                    - self.config.overlap_penalty * (ov_pen)
                    - role_pen
                    - r_pen
                )

                if score > best_score:
                    best_score = score
                    best = cand

            if best is None:
                break

            chosen.append(best)
            if best.R in r_count:
                r_count[best.R] += 1

            chosen_sets.append(set(best.dezenas))
            covered_C |= (set(best.dezenas) & set(C))

            if best.role == "contrarian" and need_contrarian > 0:
                need_contrarian -= 1
            if best.role == "aggressive" and need_aggressive > 0:
                need_aggressive -= 1

        if len(chosen) < qtd:
            # fallback: completa com os melhores que respeitam hard overlap
            for cand in candidates:
                if len(chosen) >= qtd:
                    break
                if cand.R < 8 or cand.R > 11:
                    continue
                s = set(cand.dezenas)
                if any(len(s & cs) > self.config.overlap_max for cs in chosen_sets):
                    continue
                if cand in chosen:
                    continue
                chosen.append(cand)
                if cand.R in r_count:
                    r_count[cand.R] += 1
                chosen_sets.append(s)
                covered_C |= (s & set(C))

        if len(chosen) < qtd:
            raise RuntimeError(f"Não consegui formar o portfólio com {qtd} jogos dentro do anti-overlap.")

        # Repair: garante cobertura total do complemento (10 ausentes) se ativado
        if self.config.require_full_complement_coverage:
            missing = set(C) - covered_C
            if missing:
                chosen = self._repair_complement_coverage(chosen, candidates, C)

        # retorna listas ordenadas
        return [list(c.dezenas) for c in chosen[:qtd]]

    def _repair_complement_coverage(
        self,
        chosen: List[_Candidate],
        candidates: List[_Candidate],
        C: List[int],
    ) -> List[_Candidate]:
        Cset = set(C)

        def covered(ch: List[_Candidate]) -> set[int]:
            out = set()
            for c in ch:
                out |= (set(c.dezenas) & Cset)
            return out

        ch = list(chosen)
        cov = covered(ch)
        missing = Cset - cov
        if not missing:
            return ch

        # tenta trocar 1-2 apostas (as que menos ajudam na cobertura) por candidatas que cobrem faltantes
        # mantendo hard overlap
        def overlap_ok(candidate: _Candidate, ch_sets: List[set[int]], skip_idx: int) -> bool:
            s = set(candidate.dezenas)
            for j, cs in enumerate(ch_sets):
                if j == skip_idx:
                    continue
                if len(s & cs) > self.config.overlap_max:
                    return False
            return True

        for _ in range(40):
            cov = covered(ch)
            missing = Cset - cov
            if not missing:
                return ch

            ch_sets = [set(c.dezenas) for c in ch]

            # escolhe índice pior em cobertura (menos dezenas de C)
            idxs = list(range(len(ch)))
            idxs.sort(key=lambda i: (len(set(ch[i].dezenas) & Cset), ch[i].score_base))
            worst_i = idxs[0]

            # busca melhor substituta que cubra pelo menos 1 faltante
            best_rep = None
            best_gain = -1
            for cand in candidates:
                if cand in ch:
                    continue
                s = set(cand.dezenas)
                gain = len((s & Cset) - cov)
                if gain <= 0:
                    continue
                if not overlap_ok(cand, ch_sets, worst_i):
                    continue
                # preferir maior ganho, depois score_base
                key = (gain, cand.score_base)
                if gain > best_gain or (gain == best_gain and (best_rep is None or cand.score_base > best_rep.score_base)):
                    best_rep = cand
                    best_gain = gain

            if best_rep is None:
                break

            ch[worst_i] = best_rep

        return ch

    # ------------------------- API pública -------------------------

    def gerar_lote(
        self,
        ultimo_resultado: List[int],
        qtd: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Gera um lote de apostas (Preset Mestre).

        ✅ Default: usa o otimizador de portfólio (anti-colapso)
        🔁 Fallback: mantém o gerador antigo se use_portfolio_optimizer=False.

        Determinismo:
        - O /gerar chama com seed determinístico (base -> seed).
        - Internamente, o otimizador deriva seeds estáveis para cada candidato.
        """
        if qtd is None:
            qtd = self.config.target_qtd
        qtd = int(qtd)
        if qtd <= 0:
            raise ValueError("qtd inválida para gerar_lote.")

        # Normaliza último resultado
        L = sorted(set(int(x) for x in ultimo_resultado if 1 <= int(x) <= N_UNIVERSO))
        if len(L) != N_DEZENAS:
            raise ValueError("último resultado inválido: precisa de 15 dezenas 1–25.")

        universo = list(range(1, N_UNIVERSO + 1))
        C = [n for n in universo if n not in L]

        # Seed determinístico padrão
        base_seed = int(seed) if seed is not None else deterministic_seed(L)

        if self.config.use_portfolio_optimizer:
            cands = self._generate_candidate_pool(L, C, base_seed=base_seed, qtd=qtd)
            lot = self._portfolio_greedy_select(cands, L, C, qtd=qtd)
            return [sorted(a) for a in lot]

        # -----------------------------
        # GERADOR LEGADO (compatível)
        # -----------------------------
        rng = random.Random(base_seed)

        base_R = list(self.config.base_R_plan)
        plano_R: List[int] = []
        for i in range(qtd):
            idx = i % len(base_R)
            val = base_R[idx]
            if self.alpha > 0.35:
                if val == 9 and rng.random() < (self.alpha - 0.35):
                    val = 10
                if val == 10 and rng.random() < (self.alpha - 0.35) / 2.0:
                    val = 11
            plano_R.append(val)

        apostas: List[List[int]] = []
        tentativas = 0
        max_tentativas = max(qtd * 25, 250)

        while len(apostas) < qtd and tentativas < max_tentativas:
            tentativas += 1
            idx_lote = len(apostas)
            R_target = max(6, min(13, plano_R[idx_lote]))

            from_last = self._sample_weighted(rng, L, R_target)
            from_comp = self._sample_weighted(rng, C, N_DEZENAS - len(from_last))
            aposta = sorted(set(from_last + from_comp))

            if len(aposta) < N_DEZENAS:
                faltam = N_DEZENAS - len(aposta)
                pool = [n for n in universo if n not in aposta]
                rng.shuffle(pool)
                aposta.extend(pool[:faltam])
                aposta = sorted(set(aposta))

            aposta = self._ajustar_shape_local(rng, aposta)
            if not shape_ok_mestre(aposta):
                continue

            if any(len(set(aposta) & set(b)) > self.config.overlap_max for b in apostas):
                continue
            if any(set(aposta) == set(b) for b in apostas):
                continue

            apostas.append(sorted(aposta))

        if len(apostas) < qtd:
            raise RuntimeError(
                f"Falha ao gerar lote Mestre com {qtd} apostas dentro das regras "
                f"(shape Mestre + anti-overlap≤{self.config.overlap_max}). "
                f"Gerei apenas {len(apostas)} apostas em {tentativas} tentativas."
            )

        return [sorted(a) for a in apostas]
