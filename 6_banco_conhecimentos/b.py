from typing import Dict, List, Tuple


class SmartInvestor:
    def __init__(self):
        # Armazena fatos como tuplas (nome, valor)
        # Exemplo: ('selic', 'alta')
        self.facts = set()
        # Regras armazenadas como listas de premissas e conclusões
        # Exemplo: [(['selic', 'alta'], 'investimento_tesouro_direto')]
        self.rules = []

    def add_fact(self, fact: Tuple[str, int]):
        # Adiciona um fato ao banco de conhecimentos
        self.facts.add(fact)

    def remove_fact(self, fact: tuple[str, int]):
        # Remove um fato do banco de conhecimentos
        self.facts.discard(fact)

    def remove_rule(self, premise: str, conclusion: str):
        # Remove uma regra do banco de conhecimentos
        self.rules.remove((premise, conclusion))

    def add_rule(self, premise: str, conclusion: str):
        # Adiciona uma regra ao banco de conhecimentos
        self.rules.append((premise, conclusion))

    def infer(self):
        """Utiiza o modus ponens para inferir novos fatos com base nas regras e fatos existentes."""
        # Aplica regras para inferir novos fatos
        new_inferences = True
        while new_inferences:
            # Continua inferindo até que não haja mais novos fatos
            new_inferences = False
            for premise, conclusion in self.rules:
                # Verifica se todas as premissas estão nos fatos
                if all(fact in self.facts for fact in premise):
                    # Adiciona a conclusão aos fatos se ainda não estiver presente
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        new_inferences = True

    def ask(self, query):
        # Verifica se um fato está no banco de conhecimentos
        self.infer()
        return query in self.facts


smart_investor = SmartInvestor()

##### Fatos iniciais ######
smart_investor.add_fact(('selic', 'alta'))  # porcentagem
smart_investor.add_fact(('inflacao', 'alta'))  # porcentagem
smart_investor.add_fact(('petroleo', 'alto'))  # dolar por barril
smart_investor.add_fact(('dolar', 'alto'))  # porcentagem


##### Regras de investimento######
# alta selic
smart_investor.add_rule([('selic', 'alta')], 'investimento_tesouro_direto')
# alta inflacao e selic
smart_investor.add_rule(
    [('inflacao', 'alta')], 'investimento_tesouro_ipca')
# alta petroleo
smart_investor.add_rule([('petroleo', 'alto')], 'investimento_PETR4')
# alta dolar
smart_investor.add_rule([('dolar', 'alto')], 'investimento_ativo_dolar')
# dolar e petroleo altos
smart_investor.add_rule(
    [('dolar', 'alto'), ('petroleo', 'alto')], 'investimento_petroleo_estrangeiro')


##### Consultas ######
print('Investir em Tesouro Direto?',
      smart_investor.ask('investimento_tesouro_direto'))  # True
print(smart_investor.facts)
print('Investir em Tesouro IPCA?',
      smart_investor.ask('investimento_tesouro_ipca'))  # True
print('Investir em PETR4?', smart_investor.ask('investimento_PETR4'))  # True
print('Investir em Ativo Dolar?',
      smart_investor.ask('investimento_ativo_dolar'))  # True
print('Investir em Petroleo Estrangeiro?',
      smart_investor.ask('investimento_petroleo_estrangeiro'))  # True

print('Investir em bitcoin?', smart_investor.ask(
    'investimento_bitcoin'))  # False

smart_investor.add_fact(('bitcoin', 'alto'))  # Adiciona fato sobre bitcoin
# Adiciona regra sobre bitcoin
smart_investor.add_rule([('bitcoin', 'alto')], 'investimento_bitcoin')
print('Fato bitcoin alto adicionado.')
print('Investir em bitcoin?', smart_investor.ask('investimento_bitcoin'))  # True

# Remover fato sobre selic
smart_investor.remove_fact(('selic', 'alta'))
smart_investor.remove_rule(
    [('selic', 'alta')], 'investimento_tesouro_direto')
smart_investor.remove_fact(
    ('investimento_tesouro_direto'))  # Recalcula inferências
print('Fato selic alta removido.')
print('Investir em Tesouro Direto?',
      smart_investor.ask('investimento_tesouro_direto'))  # False
