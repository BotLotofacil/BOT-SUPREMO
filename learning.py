"""
learning.py

Núcleo de aprendizado para o Oráculo da Lotofácil.

Principais características:

- Estado salvo em arquivo JSON (LEARN_STATE_PATH ou learn_state.json).
- Janela deslizante de concursos (cfg.janela).
- Aprendizado baseado em:
    * desempenho do lote (melhor, média, top-K)
    * reforço/punição por dezena conforme acertos do lote
    * memória acumulada de hits/misses e apostas boas/ruins
- Alpha dinâmico (ajusta o "peso" do aprendizado com base no desempenho).
- Bias por dezena normalizado em [-1, 1] para ser usado pelo gerador.
- Sincronização OPCIONAL com GitHub:
    * GITHUB_STATE_TOKEN  → token pessoal (PAT) com escopo de repo.
    * GITHUB_STATE_REPO   → "dono/repositorio" (ex.: "BotLotofacil/BOT-SUPREMO").
    * GITHUB_STATE_PATH   → caminho do arquivo (ex.: "data/learn_state.json").
    * GITHUB_STATE_BRANCH → branch alvo (ex.: "state" ou "main").
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List


# ----------------------------------------------------------------------
# Configuração geral do aprendizado
# ----------------------------------------------------------------------


@dataclass
class LearnConfig:
    """
    Configuração do núcleo de aprendizado.
    """

    # Quantos concursos manter no histórico (janela deslizante)
    janela: int = 80

    # Alpha base (ponto de partida e piso da adaptação)
    alpha_base: float = 0.36

    # Alvo de desempenho para o top-K (em acertos)
    alvo_topk: float = 10.0

    # Passo de ajuste de alpha em função do desvio do alvo_topk
    passo_alpha: float = 0.03

    # Caminho padrão do arquivo de estado
    state_path: str = field(
        default_factory=lambda: os.environ.get("LEARN_STATE_PATH", "learn_state.json")
    )


STATE_PATH_DEFAULT = os.environ.get("LEARN_STATE_PATH", "learn_state.json")


@dataclass
class LearnState:
    """
    Estado persistente de aprendizado.

    - alpha: fator de aprendizagem atual.
    - bias_num: viés por dezena (1..25) em float.
    - history: lista de últimos resultados oficiais conhecidos.
    - stats_num: estatísticas por dezena (hits, misses, boas/ruins).
    - meta: metadados para debug ou análise posterior.
    """

    alpha: float = 0.36
    bias_num: Dict[int, float] = field(default_factory=dict)
    history: List[List[int]] = field(default_factory=list)
    stats_num: Dict[int, Dict[str, int]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def _hits(aposta: List[int], oficial: List[int]) -> int:
    """Conta quantos acertos uma aposta teve vs. um resultado oficial."""
    return len(set(aposta) & set(oficial))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------
# Núcleo principal de aprendizado
# ----------------------------------------------------------------------


class LearningCore:
    """
    Núcleo de aprendizado leve, independente de frameworks externos de ML.

    Responsabilidades:
    - Carregar e salvar LearnState em JSON.
    - Sincronizar estado com GitHub (opcional).
    - Manter janela deslizante de resultados oficiais.
    - Atualizar bias_num com base em desempenho de lotes /confirmar.
    """

    def __init__(self, cfg: LearnConfig | None = None) -> None:
        self.cfg = cfg or LearnConfig()
        self.path = self.cfg.state_path or STATE_PATH_DEFAULT

        self.state: LearnState = LearnState(alpha=self.cfg.alpha_base)
        self._carregar_ou_inicializar()

    # ------------------------------------------------------
    # Carregamento / salvamento
    # ------------------------------------------------------

    def _carregar_ou_inicializar(self) -> None:
        """
        Tenta carregar o estado de disco; se não existir, tenta recuperar do GitHub.
        Se tudo falhar, inicializa estado padrão.
        """
        # 1) Tenta carregar de arquivo local
        if os.path.exists(self.path):
            if self._try_load_from_local():
                return

        # 2) Se não deu, tenta baixar do GitHub (se configurado)
        if self._try_load_from_github():
            # Garante que já salva localmente assim que puxar da nuvem
            self._salvar()
            return

        # 3) Estado novo
        self.state = LearnState(alpha=self.cfg.alpha_base)
        self._salvar()

    def _try_load_from_local(self) -> bool:
        """Carrega estado do arquivo local. Retorna True se deu certo."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[LearningCore] Erro ao carregar estado de '{self.path}': {e}")
            return False

        try:
            self._apply_loaded_data(data)
            return True
        except Exception as e:
            print(f"[LearningCore] Erro ao interpretar estado local: {e}")
            return False

    def _try_load_from_github(self) -> bool:
        """
        Tenta buscar o estado de aprendizado a partir do arquivo salvo no GitHub.

        Só entra aqui se as variáveis de ambiente estiverem configuradas.
        Retorna True se conseguiu carregar algum JSON válido.
        """
        token = os.environ.get("GITHUB_STATE_TOKEN", "").strip()
        repo = os.environ.get("GITHUB_STATE_REPO", "").strip()
        path = os.environ.get("GITHUB_STATE_PATH", "").strip()
        branch = os.environ.get("GITHUB_STATE_BRANCH", "main").strip() or "main"

        if not token or not repo or not path:
            return False

        api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

        try:
            import httpx

            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            with httpx.Client(timeout=10.0) as client:
                resp = client.get(api_url, headers=headers, params={"ref": branch})
                if resp.status_code != 200:
                    print(
                        f"[LearningCore] GET GitHub estado falhou "
                        f"(status={resp.status_code})"
                    )
                    return False

                body = resp.json()
                content_b64 = body.get("content")
                if not content_b64:
                    return False

                decoded = base64.b64decode(content_b64).decode("utf-8")
                data = json.loads(decoded)

                self._apply_loaded_data(data)
                print("[LearningCore] Estado recuperado do GitHub com sucesso.")
                return True
        except Exception as e:
            print(f"[LearningCore] Erro ao recuperar estado do GitHub: {e}")
            return False

    def _apply_loaded_data(self, data: Dict[str, Any]) -> None:
        """Converte o dicionário carregado em LearnState."""
        try:
            alpha = float(data.get("alpha", self.cfg.alpha_base))
        except Exception:
            alpha = self.cfg.alpha_base

        # bias_num pode vir com chaves str ou int
        bias_raw = data.get("bias_num", {})
        bias_num: Dict[int, float] = {}
        for k, v in bias_raw.items():
            try:
                ik = int(k)
                fv = float(v)
                if 1 <= ik <= 25:
                    bias_num[ik] = fv
            except Exception:
                continue

        # histórico de resultados
        history_raw = data.get("history", [])
        history: List[List[int]] = []
        for linha in history_raw:
            try:
                dezenas = [int(x) for x in linha]
                dezenas = sorted({d for d in dezenas if 1 <= d <= 25})
                if len(dezenas) >= 15:
                    history.append(dezenas)
            except Exception:
                continue

        # stats_num (opcional em estados antigos)
        stats_raw = data.get("stats_num", {})
        stats_num: Dict[int, Dict[str, int]] = {}
        for k, v in stats_raw.items():
            try:
                ik = int(k)
                if not (1 <= ik <= 25):
                    continue
                stats_num[ik] = {
                    "hits": int(v.get("hits", 0)),
                    "miss": int(v.get("miss", 0)),
                    "good": int(v.get("good", 0)),
                    "bad": int(v.get("bad", 0)),
                }
            except Exception:
                continue

        meta = data.get("meta", {})

        self.state = LearnState(
            alpha=alpha,
            bias_num=bias_num,
            history=history,
            stats_num=stats_num,
            meta=meta,
        )

        # Garante janela máxima
        if len(self.state.history) > self.cfg.janela:
            self.state.history = self.state.history[-self.cfg.janela :]

    def _salvar(self) -> None:
        """
        Salva o estado em disco e, se configurado, sincroniza com o GitHub.

        - Sempre grava um JSON local em self.path (LEARN_STATE_PATH).
        - Se as variáveis de ambiente de GitHub estiverem definidas, realiza
          um PUT via API para manter uma cópia durável no repositório.
        """
        raw = asdict(self.state)

        # Serialização amigável
        raw["bias_num"] = {int(k): float(v) for k, v in self.state.bias_num.items()}
        raw["history"] = list(self.state.history)[-self.cfg.janela :]

        # stats_num: garante ints
        stats_num_serial: Dict[int, Dict[str, int]] = {}
        for d in range(1, 26):
            st = self.state.stats_num.get(d, {})
            stats_num_serial[d] = {
                "hits": int(st.get("hits", 0)),
                "miss": int(st.get("miss", 0)),
                "good": int(st.get("good", 0)),
                "bad": int(st.get("bad", 0)),
            }
        raw["stats_num"] = stats_num_serial

        # 1) Salva localmente no container (melhor esforço)
        try:
            dirname = os.path.dirname(self.path) or "."
            os.makedirs(dirname, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Não derruba o bot se der erro de IO; apenas loga.
            print(f"[LearningCore] Erro ao salvar estado local em '{self.path}': {e}")

        # 2) Tenta sincronizar com GitHub (opcional)
        try:
            self._sync_to_github(raw)
        except Exception as e:
            # Falhas de rede/API não podem parar o fluxo principal.
            print(f"[LearningCore] Erro ao sincronizar estado no GitHub: {e}")

    # ------------------------------------------------------
    # Sincronização opcional do estado com GitHub
    # ------------------------------------------------------

    def _sync_to_github(self, raw: dict) -> None:
        """Envia o estado para um arquivo em um repositório GitHub.

        Esta função é *opcional* e só roda se todas as variáveis abaixo
        estiverem definidas no ambiente do container (Railway):

        - GITHUB_STATE_TOKEN  → token pessoal (PAT) com escopo de repo.
        - GITHUB_STATE_REPO   → "dono/repositorio" (ex.: "BotLotofacil/BOT-SUPREMO").
        - GITHUB_STATE_PATH   → caminho do arquivo (ex.: "data/learn_state.json").
        - GITHUB_STATE_BRANCH → branch alvo (ex.: "state" ou "main").

        Se algo der errado (rede, auth, etc.), o erro é logado via print
        mas *não* interrompe o fluxo do bot.
        """
        token = os.environ.get("GITHUB_STATE_TOKEN", "").strip()
        repo = os.environ.get("GITHUB_STATE_REPO", "").strip()
        path = os.environ.get("GITHUB_STATE_PATH", "").strip()
        branch = os.environ.get("GITHUB_STATE_BRANCH", "main").strip() or "main"

        if not token or not repo or not path:
            # Não configurado → nada a fazer.
            return

        api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

        # Serializa o dict em JSON e codifica em Base64 (exigência da API).
        payload_json = json.dumps(raw, ensure_ascii=False, indent=2)
        encoded = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        try:
            import httpx  # httpx já vem como dependência indireta do python-telegram-bot

            with httpx.Client(timeout=10.0) as client:
                # Primeiro, tenta obter o SHA atual do arquivo (se existir)
                sha = None
                resp_get = client.get(api_url, headers=headers, params={"ref": branch})
                if resp_get.status_code == 200:
                    try:
                        sha = resp_get.json().get("sha")
                    except Exception:
                        sha = None

                data = {
                    "message": "Atualiza estado de aprendizado (learn_state.json) via bot.",
                    "content": encoded,
                    "branch": branch,
                }
                if sha:
                    data["sha"] = sha

                resp_put = client.put(api_url, headers=headers, json=data)
                if resp_put.status_code >= 400:
                    print(
                        f"[LearningCore] Falha ao fazer PUT no GitHub "
                        f"(status={resp_put.status_code}): {resp_put.text[:200]}"
                    )
        except Exception as e:
            print(f"[LearningCore] Exceção ao sincronizar com GitHub: {e}")

    # ------------------------ API pública ------------------------

    def get_alpha(self) -> float:
        return float(self.state.alpha)

    def get_bias_num(self) -> Dict[int, float]:
        # Sempre retorna uma cópia para evitar mutação externa
        return dict(self.state.bias_num)

    # ------------------------------------------------------
    # Registro de resultados e lotes
    # ------------------------------------------------------

    def registrar_resultado_oficial(self, dezenas: List[int]) -> None:
        """
        Registra um novo resultado oficial na memória de histórico,
        respeitando a janela configurada. Também atualiza stats_num
        com hits/misses por dezena.
        """
        dezenas = sorted({d for d in dezenas if 1 <= d <= 25})
        if len(dezenas) < 15:
            return

        # Atualiza histórico
        self.state.history.append(dezenas)
        if len(self.state.history) > self.cfg.janela:
            self.state.history = self.state.history[-self.cfg.janela :]

        # Atualiza contadores por dezena (hit/miss)
        dezenas_set = set(dezenas)
        for d in range(1, 26):
            st = self.state.stats_num.setdefault(
                d, {"hits": 0, "miss": 0, "good": 0, "bad": 0}
            )
            if d in dezenas_set:
                st["hits"] += 1
            else:
                st["miss"] += 1

        self._salvar()

    def registrar_lote_gerado(
        self,
        oficial_base: List[int],
        apostas: List[List[int]],
        tag: str = "gerar",
    ) -> None:
        """
        Registra um lote que foi gerado pelo /gerar para telemetria leve.

        Aqui só guardamos meta-informações básicas;
        o aprendizado efetivo é feito em aprender_com_lote.
        """
        dezenas = sorted({d for d in oficial_base if 1 <= d <= 25})
        if len(dezenas) >= 15:
            self.registrar_resultado_oficial(dezenas)

        meta_lotes = self.state.meta.get("lotes", [])
        meta_lotes.append(
            {
                "tag": tag,
                "qtd_apostas": len(apostas),
                "base": dezenas,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.state.meta["lotes"] = meta_lotes[-200:]
        self._salvar()

    # ------------------------------------------------------
    # Aprendizado principal via /confirmar
    # ------------------------------------------------------

    def aprender_com_lote(
        self,
        oficial: List[int],
        apostas: List[List[int]],
        tag: str = "confirmar",
    ) -> Dict[str, Any]:
        """
        Aplica aprendizado sobre um lote de apostas dado um resultado oficial.

        Retorna um dicionário com:
        - melhor: melhor número de acertos.
        - media: média de acertos do lote.
        - topk: média dos K melhores (K adaptativo).
        - placares: lista de acertos aposta a aposta.
        - alpha: novo alpha.
        - lote_bom: bool indicando se o lote foi forte.
        """
        oficial = sorted({d for d in oficial if 1 <= d <= 25})
        if len(oficial) < 15:
            raise ValueError("Resultado oficial inválido para aprendizado.")

        if not apostas:
            raise ValueError("Nenhuma aposta fornecida para aprendizado.")

        # Garante que o histórico esteja atualizado + stats hit/miss
        self.registrar_resultado_oficial(oficial)

        # ---------------- Desempenho bruto do lote ----------------
        placares: List[int] = [_hits(a, oficial) for a in apostas]
        melhor = max(placares)
        media = sum(placares) / len(placares)

        k = max(1, min(3, len(apostas) // 3 or 1))
        topk_vals = sorted(placares, reverse=True)[:k]
        topk = sum(topk_vals) / len(topk_vals)

        # ---------------- Atualização de alpha ----------------
        alvo = self.cfg.alvo_topk
        delta = topk - alvo
        alpha_old = float(self.state.alpha)
        alpha_new = alpha_old + self.cfg.passo_alpha * delta
        alpha_new = _clamp(alpha_new, 0.10, 0.80)
        self.state.alpha = alpha_new

        # ---------------- Reforço / punição por dezena ----------------
        dezenas_oficial = set(oficial)

        # recompensa acumulada por dezena neste lote
        reward: Dict[int, float] = {d: 0.0 for d in range(1, 26)}

        # 1) Reforço direto: dezenas que caíram no resultado
        for d in dezenas_oficial:
            reward[d] += 2.0  # bônus forte para acertos oficiais

        # 2) Apostas boas/ruins influenciam stats_num e reward
        for aposta, score in zip(apostas, placares):
            dez_set = set(aposta)

            # Define um peso para a aposta em função do número de acertos
            if score >= 12:
                peso = +2.0
            elif score == 11:
                peso = +1.2
            elif score == 10:
                peso = +0.8
            elif score == 9:
                peso = +0.4
            elif score == 8:
                peso = 0.0  # neutro
            elif score == 7:
                peso = -0.4
            else:  # 0–6
                peso = -0.7

            # Atualiza reward por dezena participante desta aposta
            for d in dez_set:
                if 1 <= d <= 25:
                    reward[d] += peso

            # Atualiza stats boas/ruins por dezena
            is_good = score >= 11
            is_bad = score <= 7
            for d in dez_set:
                if 1 <= d <= 25:
                    st = self.state.stats_num.setdefault(
                        d, {"hits": 0, "miss": 0, "good": 0, "bad": 0}
                    )
                    if is_good:
                        st["good"] += 1
                    if is_bad:
                        st["bad"] += 1

        # ---------------- Converte reward + stats em novo bias ----------------
        bias = dict(self.state.bias_num)
        eta = 0.5 * alpha_new  # taxa de aprendizado efetiva

        # Evita divisor zero
        total_concursos = max(1, len(self.state.history))

        for d in range(1, 26):
            st = self.state.stats_num.setdefault(
                d, {"hits": 0, "miss": 0, "good": 0, "bad": 0}
            )

            # Taxa de acerto global (hit-rate)
            hit_rate = st["hits"] / total_concursos

            # Desempenho em apostas boas/ruins
            good_bad_total = max(1, st["good"] + st["bad"])
            good_rate = st["good"] / good_bad_total

            # Score combinado desta dezena
            #   reward_lote  → efeito imediato do último concurso
            #   hit_rate     → tendência de aparecer nos oficiais
            #   good_rate    → tendência de aparecer nas apostas boas
            combined = (
                reward[d]
                + 4.0 * hit_rate  # peso moderado
                + 3.0 * good_rate
            )

            old = bias.get(d, 0.0)
            new = (1.0 - eta) * old + eta * combined
            bias[d] = float(new)

        # Normaliza bias para ficar aproximadamente em [-1, 1]
        values = list(bias.values()) or [0.0]
        media_bias = sum(values) / len(values)
        for d in bias:
            bias[d] -= media_bias

        max_abs = max((abs(v) for v in bias.values()), default=0.0)
        if max_abs > 0:
            fator = 1.0 / max_abs
            for d in bias:
                bias[d] = float(bias[d] * fator)

        self.state.bias_num = bias

        # ---------------- Meta / Telemetria ----------------
        meta_aprend = self.state.meta.get("aprendizado", [])
        meta_aprend.append(
            {
                "tag": tag,
                "oficial": oficial,
                "placares": placares,
                "melhor": melhor,
                "media": media,
                "topk": topk,
                "alpha_before": alpha_old,
                "alpha_after": alpha_new,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.state.meta["aprendizado"] = meta_aprend[-200:]

        self._salvar()

        lote_bom = topk >= self.cfg.alvo_topk

        return {
            "melhor": melhor,
            "media": media,
            "topk": topk,
            "placares": placares,
            "alpha": alpha_new,
            "lote_bom": lote_bom,
        }
