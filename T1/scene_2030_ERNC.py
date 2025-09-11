#%% Importar librerías
import pandas as pd
import highspy as hgs
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

tolerancia = 0.00001
perdida = 0.04

#%%

# Leer el archivo CSV
df_centrales_ex = pd.read_csv('centrales_ex.csv')
df_centrales_ex = df_centrales_ex.fillna(0)

df_centrales_nuevas = pd.read_csv('centrales_n.csv')
df_centrales_nuevas = df_centrales_nuevas.fillna(0)

#%%

param_centrales = df_centrales_ex.set_index(['planta_n'])
param_centrales_nuevas = df_centrales_nuevas.set_index(['planta_n'])

#%%

# Tipos de Centrales
t_centrales = ['biomasa', 'carbon','cc-gnl', 'petroleo_diesel', 'eolica','solar', 'geotermia']

# Centrales renovables
t_ernc = ['eolica','solar', 'geotermia','minihidro', 'hidro_conv', 'biomasa']

# Para la modelación de la hidroelectricidad
dispnibilidad_hidro = [0.8215,0.6297,0.561]

# Costo de falla
costo_falla = 505.5 # mills/kWh = USD/MWh

# Años hacia el 2030
year = 2030 - 2016

#%%


index_plantas = param_centrales.index.tolist()
index_plantas_nuevas = param_centrales_nuevas.index.tolist()
dic_bloques = {'bloque_1': {'duracion': 12000 , 'demanda' : 10033.21303},
               'bloque_2': {'duracion': 4152 , 'demanda' : 7717.657157}, 
               'bloque_3': {'duracion': 3408 , 'demanda' : 6174.384447}}

#%%

# MODELO
model = pyo.ConcreteModel()

# CONJUNTOS
model.CENTRALES = pyo.Set(initialize=index_plantas)
model.CENTRALES_NUEVAS = pyo.Set(initialize=index_plantas_nuevas)
model.BLOQUES = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])

# PARAMETROS
model.param_centrales = pyo.Param(model.CENTRALES,
                                  initialize=param_centrales.to_dict(orient='index'),
                                  within=pyo.Any)

model.param_centrales_nuevas = pyo.Param(model.CENTRALES_NUEVAS,
                                  initialize=param_centrales_nuevas.to_dict(orient='index'),
                                  within=pyo.Any)

model.param_bloques = pyo.Param(model.BLOQUES, initialize=dic_bloques, within=pyo.Any)

# VARIABLES

model.generacion_ex = pyo.Var(model.CENTRALES, model.BLOQUES, within=pyo.NonNegativeReals)
model.generacion_nuevas = pyo.Var(model.CENTRALES_NUEVAS, model.BLOQUES, within=pyo.NonNegativeReals)
model.potencia_in_nuevas = pyo.Var(model.CENTRALES_NUEVAS, within=pyo.NonNegativeReals)
model.falla = pyo.Var(model.BLOQUES, within=pyo.NonNegativeReals)

#%%


# RESTRICCIONES

########### DEMANDA ###########
def fd_hidro(bloque):
    return dispnibilidad_hidro[0] if bloque == 'bloque_1' else \
           dispnibilidad_hidro[1] if bloque == 'bloque_2' else \
           dispnibilidad_hidro[2]

def balance_demanda(model, bloque):
    # Energía neta generada por existentes y nuevas (sin disponibilidad ni eficiencia)
    gen_ex   = sum(model.generacion_ex[planta, bloque]   for planta in model.CENTRALES)
    gen_new  = sum(model.generacion_nuevas[planta, bloque] for planta in model.CENTRALES_NUEVAS)

    return (gen_ex + gen_new + model.falla[bloque]) * (1/(1+perdida)) \
       >= model.param_bloques[bloque]['demanda'] * model.param_bloques[bloque]['duracion']


############ MÁXIMA GENERACIÓN ###########

def max_gen_ex(model, planta, bloque):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['hidro', 'hidro_conv', 'minihidro']:
        disp = fd_hidro(bloque)
    else:
        disp = model.param_centrales[planta]['disponibilidad']
    return model.generacion_ex[planta, bloque] \
           <= model.param_centrales[planta]['potencia_neta_mw'] * model.param_bloques[bloque]['duracion'] * disp

def max_gen_nuevas(model, planta, bloque):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['hidro', 'hidro_conv', 'minihidro']:
        disp = fd_hidro(bloque)
    else:
        disp = model.param_centrales_nuevas[planta]['disponibilidad']
    return model.generacion_nuevas[planta, bloque] \
           <= model.potencia_in_nuevas[planta] * model.param_bloques[bloque]['duracion'] * disp

############ MÁXIMA CAPACIDAD INSTALADA NUEVAS ###########
def max_capacidad_nuevas(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in t_ernc:
        limite = model.param_centrales_nuevas[planta]['maxima_restriccion_2030_MW']
        # Si quieres que “0” signifique “sin límite”, puedes saltarte solo si limite <= 0
        return model.potencia_in_nuevas[planta] <= limite if limite > 0 else pyo.Constraint.Skip
    else:
        return pyo.Constraint.Skip

def anualidad(r, n):
    return r / (1 - (1 + r)**(-n))


############ META ENERGIAS RENOVABLES ###########
def meta_ernc_rule(model):
    # energía total (existentes + nuevas)
    gen_total = (
        sum(model.generacion_ex[planta,bloque]   for planta in model.CENTRALES        for bloque in model.BLOQUES) +
        sum(model.generacion_nuevas[planta,bloque] for planta in model.CENTRALES_NUEVAS for bloque in model.BLOQUES)
    )

    # energía ERNC (existentes + nuevas)
    gen_ernc = (
        sum(model.generacion_ex[planta,bloque]     for planta in model.CENTRALES
                                         if model.param_centrales[planta]['tecnologia'] in t_ernc
                                         for bloque in model.BLOQUES) +
        sum(model.generacion_nuevas[planta,bloque] for planta in model.CENTRALES_NUEVAS
                                         if model.param_centrales_nuevas[planta]['tecnologia'] in t_ernc
                                         for bloque in model.BLOQUES)
    )

    return gen_ernc >= 0.30 * gen_total


# Restricciones al modelo
model.demanda_constraint = pyo.Constraint(model.BLOQUES, rule=balance_demanda)
model.max_gen_constraint = pyo.Constraint(model.CENTRALES, model.BLOQUES, rule=max_gen_ex)
model.max_gen_nuevas_constraint = pyo.Constraint(model.CENTRALES_NUEVAS, model.BLOQUES, rule=max_gen_nuevas)
model.max_capacidad_nuevas_constraint = pyo.Constraint(model.CENTRALES_NUEVAS, rule=max_capacidad_nuevas)
model.meta_ernc = pyo.Constraint(rule=meta_ernc_rule)

# FUNCION OBJETIVO

# operación EXISTENTES
model.op_ex = pyo.Expression(
    expr=sum(
        model.generacion_ex[planta, bloque] *
        (model.param_centrales[planta]['costo_variable_nc'] +
         model.param_centrales[planta]['costo_variable_t'])
        for planta in model.CENTRALES
        for bloque in model.BLOQUES
    )
)

# operación NUEVAS
model.op_new = pyo.Expression(
    expr=sum(
        model.generacion_nuevas[planta, bloque] *
        (model.param_centrales_nuevas[planta]['cvnc_usd_MWh'] +
         model.param_centrales_nuevas[planta]['linea_peaje_usd_MWh'])
        for planta in model.CENTRALES_NUEVAS
        for bloque in model.BLOQUES
    )
)

# inversión NUEVAS (MW→kW * anualidad * CAPEX)
model.inv_new = pyo.Expression(
    expr=sum(
        model.potencia_in_nuevas[planta] * 1000 *
        anualidad(model.param_centrales_nuevas[planta]['tasa_descuento'],
                  model.param_centrales_nuevas[planta]['vida_util_anos']) *
        model.param_centrales_nuevas[planta]['inversion_usd_kW_neto']
        for planta in model.CENTRALES_NUEVAS
    )
)

# costo FALLAS
model.costo_fallas = pyo.Expression(
    expr=sum(model.falla[bloque] * costo_falla for bloque in model.BLOQUES)
)

# FUNCION OBJETIVO FINAL
model.obj = pyo.Objective(
    expr = model.op_ex + model.op_new + model.inv_new + model.costo_fallas,
    sense = pyo.minimize
)

# Si quieres traer todo a valor presente 2016, usa esto:
""" 
# tasa para traer 2030 a 2016 (elige la de la pauta)
r_df = 0.05  # ej.

# factor de descuento de 2016 a 2030
df_2016_2030 = 1 / (1 + r_df)**(2030 - 2016)

# si ya definiste:
# model.op_ex, model.op_new, model.inv_new, model.costo_fallas (pyo.Expression)

model.obj = pyo.Objective(
    expr = df_2016_2030 * (model.op_ex + model.op_new + model.inv_new + model.costo_fallas),
    sense = pyo.minimize
)
 """

#%%

# Resolver el modelo
solver = pyo.SolverFactory('highs')

solver.options['mip_rel_gap'] = tolerancia

results = solver.solve(model, tee=True)

# Ver resultados
print(f"Status: {results}")

# %%

# Ver los resultados de la generación
for planta in model.CENTRALES:
    for bloque in model.BLOQUES:
        print(f'Generación de {planta} en {bloque}: {model.generacion[planta, bloque].value}')



# %%
