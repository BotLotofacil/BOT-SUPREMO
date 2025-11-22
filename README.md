# Bot Lotofácil – Oráculo Supremo

Bot Telegram com geração determinística e aprendizado real.
Comando principal: /gerar

Estrutura:
- bot.py — núcleo Telegram + segurança + autorização
- engine_oraculo.py — gerador supremo
- learning.py — módulo oficial de aprendizado
- data/history.csv — 500 últimos resultados
- whitelist.txt — lista de usuários autorizados

Requisitos:
pip install -r requirements.txt

Execução:
python bot.py
