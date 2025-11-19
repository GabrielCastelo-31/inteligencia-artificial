from itertools import product
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

INFLUENZA_PROB = 0.08
TUBERCULOSE_PROB = 0.0004
# --- Probabilidades (P(sintoma=1 | doenca=1)) a partir de estudos ---
P_TOSSE_INFLUENZA = 0.93
P_FEBRE_INFLUENZA = 0.69
P_MIALGIA_INFLUENZA = 0.70
P_CONGESTAO_INFLUENZA = 0.84

P_TOSSE_TB = 0.72          # cenário clínico
P_PERDA_PESO_TB = 0.49
P_FEBRE_TB = 0.41
P_SUDORESE_TB = 0.37

# RX: "qualquer alteração" para TB; pneumonia (~29%) para influenza
P_RAIOX_TB = 0.95
P_RAIOX_INFLUENZA = 0.29

# --- "leaks" (P(sintoma=1 | nenhuma doença)) ---
LEAK_TOSSE = 0.12           # tosse na população (resfriados, etc.)
LEAK_FEBRE = 0.07
LEAK_MIALGIA = 0.08
LEAK_CONGESTAO = 0.10
LEAK_PERDA_PESO = 0.05
LEAK_SUDORESE = 0.03
LEAK_RAIOX = 0.11           # falso-positivo RX p/ “qualquer alteração”


def build_noisy_or(values_when_parent_true: list[float], leak: float) -> list[list[float]]:
    """
    values_when_parent_true: lista de p(ativar) para cada pai(na ordem de 'evidence')
    leak: p(ativar) quando nenhum pai verdadeiro
    retorna(p0_row, p1_row) para TabularCPD com evidence_card = [2, ..., 2]
    """

    # p(não ativar) por pai
    q_parents = [1 - p for p in values_when_parent_true]
    q0 = 1 - leak  # p(não ativar) do leak
    cols = []  # armazena (p0, p1) por coluna
    # Gera combinações de estados dos pais na ordem binária (0/1)
    for parent_states in product([0, 1], repeat=len(values_when_parent_true)):
        q = q0  # começa com p(não ativar) do leak
        for on, qp in zip(parent_states, q_parents):
            if on == 1:
                # multiplica p(não ativar) dos pais ligados
                q *= qp
        p_on = 1 - q
        cols.append((1 - p_on, p_on))  # (p0, p1) por coluna
    # Transpõe para linhas: primeira linha p0, segunda p1
    p0_row = [c[0] for c in cols]
    p1_row = [c[1] for c in cols]
    return [p0_row, p1_row]


# Definindo a estrutura da rede bayesiana
model = DiscreteBayesianNetwork([
    ('influenza', 'tosse'),
    ('influenza', 'febre'),
    ('influenza', 'mialgia'),
    ('influenza', 'congestao_nasal'),
    ('tuberculose', 'tosse'),
    ('tuberculose', 'perda_peso'),
    ('tuberculose', 'febre'),
    ('tuberculose', 'sudorese_noturna'),
    ('influenza', 'raiox_alterado'),
    ('tuberculose', 'raiox_alterado')
])
# Definindo as distribuições condicionais de probabilidade (CPDs)
influenza_cpd = TabularCPD(
    variable='influenza',
    variable_card=2,
    # Probabilidade de não ter influenza (0) e ter influenza (1)
    values=[[1-INFLUENZA_PROB], [INFLUENZA_PROB]]
)
tuberculose_cpd = TabularCPD(
    variable='tuberculose',
    variable_card=2,
    # Probabilidade de não ter tuberculose (0) e ter tuberculose (
    values=[[1-TUBERCULOSE_PROB], [TUBERCULOSE_PROB]]
)
# --- Substitua sua CPD de tosse (que estava placeholder) por Noisy-OR com {influenza, tuberculose} ---
tosse_values = build_noisy_or(
    values_when_parent_true=[P_TOSSE_INFLUENZA, P_TOSSE_TB],
    leak=LEAK_TOSSE
)

# TOSSE tem dois pais: {influenza, tuberculose}
tosse_cpd = TabularCPD(
    variable='tosse', variable_card=2,
    evidence=['influenza', 'tuberculose'], evidence_card=[2, 2],
    values=tosse_values
)

febre_values = build_noisy_or(
    values_when_parent_true=[P_FEBRE_INFLUENZA, P_FEBRE_TB],
    leak=LEAK_FEBRE
)

# FEBRE tem dois pais: {influenza, tuberculose}
febre_cpd = TabularCPD(
    variable='febre', variable_card=2,
    evidence=['influenza', 'tuberculose'], evidence_card=[2, 2],
    values=febre_values
)

# MIALGIA tem apenas influenza como pai
mialgia_cpd = TabularCPD(
    variable='mialgia', variable_card=2,
    evidence=['influenza'], evidence_card=[2],
    values=[
        # p(mialgia=0|influenza=0/1)
        [1 - LEAK_MIALGIA, 1 - P_MIALGIA_INFLUENZA],
        # p(mialgia=1|influenza=0/1)
        [LEAK_MIALGIA,     P_MIALGIA_INFLUENZA],
    ]
)

# Congestão nasal tem apenas influenza como pai
congestao_cpd = TabularCPD(
    variable='congestao_nasal', variable_card=2,
    evidence=['influenza'], evidence_card=[2],
    values=[
        [1 - LEAK_CONGESTAO, 1 - P_CONGESTAO_INFLUENZA],
        [LEAK_CONGESTAO,     P_CONGESTAO_INFLUENZA],
    ]
)

# Perda de peso tem apenas tuberculose como pai
perda_peso_cpd = TabularCPD(
    variable='perda_peso', variable_card=2,
    evidence=['tuberculose'], evidence_card=[2],
    values=[
        [1 - LEAK_PERDA_PESO, 1 - P_PERDA_PESO_TB],
        [LEAK_PERDA_PESO,     P_PERDA_PESO_TB],
    ]
)
# Sudorese noturna tem apenas tuberculose como pai
sudorese_cpd = TabularCPD(
    variable='sudorese_noturna', variable_card=2,
    evidence=['tuberculose'], evidence_card=[2],
    values=[
        [1 - LEAK_SUDORESE, 1 - P_SUDORESE_TB],
        [LEAK_SUDORESE,     P_SUDORESE_TB],
    ]
)

raiox_values = build_noisy_or(
    values_when_parent_true=[P_RAIOX_INFLUENZA, P_RAIOX_TB],
    leak=LEAK_RAIOX
)
# Raio-X alterado tem dois pais: {influenza, tuberculose}
raiox_cpd = TabularCPD(
    variable='raiox_alterado', variable_card=2,
    evidence=['influenza', 'tuberculose'], evidence_card=[2, 2],
    values=raiox_values
)

# Adiciona as CPDs ao modelo
model.add_cpds(
    influenza_cpd,
    tuberculose_cpd,
    tosse_cpd,
    febre_cpd,
    mialgia_cpd,
    congestao_cpd,
    perda_peso_cpd,
    sudorese_cpd,
    raiox_cpd
)

# Verifica consistência
assert model.check_model(), "Modelo inválido (CPDs inconsistentes)."

# --- variável de inferência ---
infer = VariableElimination(model)


def consulta_posterior(evid_dict):
    """Imprime P(influenza=1 | evid) e P(tuberculose=1 | evid)"""
    q_inf = infer.query(['influenza'], evidence=evid_dict, show_progress=False)
    q_tb = infer.query(['tuberculose'], evidence=evid_dict,
                       show_progress=False)
    print(f"Evidência: {evid_dict}")
    print("P(influenza=1 | evid) =", round(float(q_inf.values[1]), 4))
    print("P(tuberculose=1 | evid) =", round(float(q_tb.values[1]), 6))
    print("-"*50)


# ----Cenários de teste -----
consulta_posterior({'tosse': 1, 'febre': 1})  # paciente com tosse e febre
# tosse, febre e RX alterado
consulta_posterior({'tosse': 1, 'febre': 1, 'raiox_alterado': 1})
consulta_posterior({'mialgia': 1, 'congestao_nasal': 1}
                   )  # sintomas típicos de influenza
consulta_posterior(
    # Sintomas típicos de tuberculose
    {'perda_peso': 1, 'sudorese_noturna': 1, 'raiox_alterado': 1})
consulta_posterior(
    # todos os sintomas de tuberculose
    {'tosse': 1, 'febre': 1, 'sudorese_noturna': 1, 'raiox_alterado': 1, 'perda_peso': 1})
consulta_posterior(
    # todos os sintomas das duas doenças
    {'tosse': 1, 'febre': 1, 'mialgia': 1, 'congestao_nasal': 1,
     'perda_peso': 1, 'sudorese_noturna': 1, 'raiox_alterado': 1}
)

"""
REFERÊNCIAS (links diretos) — bases usadas para montar as CPDs
----------------------------------------------------------------

CPDs de sintomas (P[sintoma=1 | doença=1]):
    - Influenza (ambulatorial, RT-PCR):
    VanWormer JJ et al., BMC Infectious Diseases, 2014.
    https://bmcinfectdis.biomedcentral.com/articles/10.1186/1471-2334-14-231
    PubMed: https://pubmed.ncbi.nlm.nih.gov/24884932/

    - Tuberculose pulmonar (coorte populacional/guia “clínico”):
    Miller LG et al., Clinical Infectious Diseases, 2000.
    https://academic.oup.com/cid/article/30/2/293/379220
    PubMed: https://pubmed.ncbi.nlm.nih.gov/10671331/

    Raio-X (P[raiox_alterado=1 | doença]):
    - TB — CXR screening (qualquer anormalidade vs. sugestivo de TB):
    WHO TB Knowledge Sharing Platform — Section 3.1.2
    https://tbksp.who.int/en/node/1407

    - Influenza — pneumonia em RX (adultos hospitalizados):
    Garg S et al., BMC Infectious Diseases, 2015.
    https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-015-1004-y
    PubMed: https://pubmed.ncbi.nlm.nih.gov/26307108/

A priori (P[influenza], P[tuberculose]):
    - Influenza — taxa de ataque sazonal (adultos 5–10%):
    WHO: https://cdn.who.int/media/docs/default-source/immunization/vpd_surveillance/vpd-surveillance-standards-publication/who-surveillancevaccinepreventable-09-influenza-r2.pdf

    - Tuberculose — Brasil:
    WHO Data – Brazil (076): https://data.who.int/countries/076
"""
