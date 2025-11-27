"""
learning.py

Núcleo de aprendizado leve para o Oráculo da Lotofácil.

Agora com:
- Estado salvo em arquivo JSON (LEARN_STATE_PATH).
- Janela deslizante de concursos (cfg.janela).
- Funções para registrar lotes gerados e aprender com o resultado oficial.
- Sincronização OPCIONAL com GitHub (se variáveis de ambiente forem configuradas).

Versão ajustada para aprendizado um pouco mais AGRESSIVO:
- Alpha reage mais ao desempenho do lote.
- Critério de alvo de desempenho ligeiramente relaxado.
- Bias por dezena mais seletivo (reforça forte quem foi muito bem,
  enfraquece forte quem foi muito mal, quase neutro para apostas medianas).
"""

import json
import os
import base64
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple
from datetime import datetime


# ----------------------------------------------------------------------
# Configuração geral do aprendizado
# ----------------------------------------------------------------------


@dataclass
class LearnConfig:
    """
    Configuração do núcleo de aprendizado.
    """

    # Janela de concursos a considerar (ex.: 80 últimos resultados)
    janela: int = 80

    # Fator base de aprendizado (quão forte puxamos o bias)
    alpha_base: float = 0.36

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
    - bias_num: viés por dezena (0..25) em float.
    - history: lista de últimos resultados oficiais conhecidos.
    - meta: metadados para debug ou análise posterior.
    """

    alpha: float = 0.36
    bias_num: Dict[int, float] = field(default_factory=dict)
    history: List[List[int]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _hits(aposta: List[int], oficial: List[int]) -> int:
    """Conta quantos acertos uma aposta teve vs. um resultado oficial."""
    return len(set(aposta) & set(oficial))


# ----------------------------------------------------------------------
# Núcleo principal de aprendizado
# ----------------------------------------------------------------------


class LearningCore:
    """
    Núcleo de aprendizado leve, sem depender de frameworks externos de ML.

    Responsabilidades:
    - Carregar e salvar LearnState em JSON.
    - M ant er janela deslizante de resultados oficiais.
    - Atualizar bias_num com base em desempenho de lotes /confirmar.
    """

    def __init__(self, cfg: LearnConfig | None = None) -> None:
        self.cfg = cfg or LearnConfig()
        self.path = self.cfg.state_path or STATE_PATH_DEFAULT

        self.state: LearnState = LearnState()
        self._carregar_ou_inicializar()

    # ------------------------------------------------------
    # Carregamento / salvamento
    # ------------------------------------------------------

    def _carregar_ou_inicializar(self) -> None:
        """
        Tenta carregar o estado de disco; se não existir, inicializa padrão.
        """
        if not os.path.exists(self.path):
            # Estado novo
            self.state = LearnState(alpha=self.cfg.alpha_base)
            self._salvar()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[LearningCore] Erro ao carregar estado de '{self.path}': {e}")
            self.state = LearnState(alpha=self.cfg.alpha_base)
            self._salvar()
            return

        try:
            alpha = float(data.get("alpha", self.cfg.alpha_base))
        except Exception:
            alpha = self.cfg.alpha_base

        bias_raw = data.get("bias_num", {})
        bias_num: Dict[int, float] = {}
        for k, v in bias_raw.items():
            try:
                ik = int(k)
                fv = float(v)
                bias_num[ik] = fv
            except Exception:
                continue

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

        meta = data.get("meta", {})

        self.state = LearnState(alpha=alpha, bias_num=bias_num, history=history, meta=meta)

        # Garante janela máxima
        if len(self.state.history) > self.cfg.janela:
            self.state.history = self.state.history[-self.cfg.janela :]
            self._salvar()

    def _salvar(self) -> None:
        """Salva o estado em disco e, se configurado, sincroniza com o GitHub.

        - Sempre grava um JSON local em self.path (LEARN_STATE_PATH).
        - Se as variáveis de ambiente de GitHub estiverem definidas, realiza
          um PUT via API para manter uma cópia durável no repositório.
        """
        raw = asdict(self.state)
        raw["bias_num"] = {int(k): float(v) for k, v in self.state.bias_num.items()}
        raw["history"] = list(self.state.history)[-self.cfg.janela :]

        # 1) Salva localmente no container (melhor esforço)
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
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
        - GITHUB_STATE_BRANCH → branch alvo (ex.: "main") opcional, default="main".

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
                        f"(status={resp_put.status_code}): {resp_put.text[:200]}",
                    )
        except Exception as e:
            print(f"[LearningCore] Exceção ao sincronizar com GitHub: {e}")

    # ------------------------ API pública ------------------------

    def get_alpha(self) -> float:
        return float(self.state.alpha)

    def get_bias_num(self) -> Dict[int, float]:
        return dict(self.state.bias_num)

    def registrar_resultado_oficial(self, dezenas: List[int]) -> None:
        """
        Registra um novo resultado oficial na memória de histórico,
        respeitando a janela configurada.
        """
        dezenas = sorted({d for d in dezenas if 1 <= d <= 25})
        if len(dezenas) < 15:
            return

        self.state.history.append(dezenas)
        if len(self.state.history) > self.cfg.janela:
            self.state.history = self.state.history[-self.cfg.janela :]
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

        meta = self.state.meta.get("lotes", [])
        meta.append(
            {
                "tag": tag,
                "qtd_apostas": len(apostas),
                "base": dezenas,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.state.meta["lotes"] = meta[-200:]
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

        # Garante que o histórico esteja atualizado
        self.registrar_resultado_oficial(oficial)

        placares: List[int] = [_hits(a, oficial) for a in apostas]
        melhor = max(placares)
        media = sum(placares) / len(placares)

        # K adaptativo (top 1 a top 3)
        k = max(1, min(3, len(apostas) // 3))
        topk_vals = sorted(placares, reverse=True)[:k]
        topk = sum(topk_vals) / len(topk_vals)

        # ------------------------------------------------------------------
        # Ajuste de alpha (um pouco mais agressivo que a versão anterior)
        # ------------------------------------------------------------------
        # Alvo ligeiramente abaixo de 10 para reconhecer mais lotes bons.
        alvo_topk = 9.8
        delta = topk - alvo_topk

        alpha_old = float(self.state.alpha)

        # Antes: 0.02 * delta → agora 0.04 * delta (responde mais rápido).
        alpha_step = 0.04
        alpha_new = alpha_old + alpha_step * delta

        # Mantém alpha em faixa segura
        alpha_new = max(0.10, min(0.80, alpha_new))
        self.state.alpha = alpha_new

        dezenas_oficial = set(oficial)

        # ------------------------------------------------------------------
        # Atualização do bias por dezena (mais seletivo/agressivo)
        # ------------------------------------------------------------------
        bias = dict(self.state.bias_num)

        for aposta, acertos in zip(apostas, placares):
            dez = set(aposta)

            # Regras:
            # - 12+ acertos: reforço máximo
            # - 11 acertos: reforço forte
            # - 10 acertos: reforço moderado
            # - 8–9 acertos: praticamente neutro
            # - 6–7 acertos: enfraquecimento moderado
            # - 0–5 acertos: enfraquecimento forte
            if acertos >= 12:
                fator = +1.2
            elif acertos == 11:
                fator = +0.9
            elif acertos == 10:
                fator = +0.4
            elif 8 <= acertos <= 9:
                fator = 0.0
            elif 6 <= acertos <= 7:
                fator = -0.4
            else:  # acertos <= 5
                fator = -0.9

            # Aplica esse fator ponderado pelo alpha atual
            for d in dez:
                if 1 <= d <= 25:
                    bias[d] = bias.get(d, 0.0) + fator * alpha_new

        # Normalização para manter os valores em [-1, 1]
        max_abs = max((abs(v) for v in bias.values()), default=0.0)
        if max_abs > 0:
            coef = 1.0 / max_abs
            for d in list(bias.keys()):
                bias[d] = float(bias[d] * coef)

        self.state.bias_num = bias

        # ------------------------------------------------------------------
        # Registro de telemetria do aprendizado
        # ------------------------------------------------------------------
        meta = self.state.meta.get("aprendizado", [])
        meta.append(
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
        self.state.meta["aprendizado"] = meta[-200:]

        self._salvar()

        lote_bom = topk >= alvo_topk

        return {
            "melhor": melhor,
            "media": media,
            "topk": topk,
            "placares": placares,
            "alpha": alpha_new,
            "lote_bom": lote_bom,
        }
