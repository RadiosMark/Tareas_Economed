#%% Importar librerías
import pandas as pd
import highspy as hgs
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

tolerancia = 0.00001
perdida = 0.004

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
t_ernc = ['eolica','solar', 'geotermia','minihidro', 'hidro_conv']

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

model = pyo.ConcreteModel()

# Definir un conjunto simple
model.CENTRALES = pyo.Set(initialize=index_plantas)
model.CENTRALES_NUEVAS = pyo.Set(initialize=index_plantas_nuevas)
model.BLOQUES = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])

model.param_centrales = pyo.Param(model.CENTRALES,
                                  initialize=param_centrales.to_dict(orient='index'),
                                  within=pyo.Any)

model.param_centrales_nuevas = pyo.Param(model.CENTRALES_NUEVAS,
                                  initialize=param_centrales_nuevas.to_dict(orient='index'),
                                  within=pyo.Any)

model.param_bloques = pyo.Param(model.BLOQUES, initialize=dic_bloques, within=pyo.Any)

model.generacion_ex = pyo.Var(model.CENTRALES, model.BLOQUES, within=pyo.NonNegativeReals)
model.generacion_nuevas = pyo.Var(model.CENTRALES_NUEVAS, model.BLOQUES, within=pyo.NonNegativeReals)
model.potencia_in_nuevas = pyo.Var(model.CENTRALES_NUEVAS, within=pyo.NonNegativeReals)
model.falla = pyo.Var(model.BLOQUES, within=pyo.NonNegativeReals)


# Funciones de restricción
def balance_demanda(model, bloque):
    sum_centrales_ex = 0
    sum_centrales_nuevas = 0

    for planta in model.CENTRALES:
        if planta in t_centrales:
            # si el profe se digna a poner bien las weas esta linea se cambia por otra cosa, no se estoy cansado
            sum_centrales_ex += model.generacion[planta, bloque] * model.param_centrales[planta]['eficiencia']
        else:
            if bloque == 'bloque_1':
                sum_centrales_ex += model.generacion[planta, bloque] * dispnibilidad_hidro[0]
            elif bloque == 'bloque_2':
                sum_centrales_ex += model.generacion[planta, bloque] * dispnibilidad_hidro[1]
            elif bloque == 'bloque_3':
                sum_centrales_ex += model.generacion[planta, bloque] * dispnibilidad_hidro[2]
    
    for planta in model.CENTRALES_NUEVAS:
        if planta in t_centrales:
            sum_centrales_nuevas += model.generacion_nuevas[planta, bloque] * model.param_centrales_nuevas[planta]['eficiencia']
        else:
            if bloque == 'bloque_1':
                sum_centrales_nuevas += model.generacion_nuevas[planta, bloque] * dispnibilidad_hidro[0]
            elif bloque == 'bloque_2':
                sum_centrales_nuevas += model.generacion_nuevas[planta, bloque] * dispnibilidad_hidro[1]
            elif bloque == 'bloque_3':
                sum_centrales_nuevas += model.generacion_nuevas[planta, bloque] * dispnibilidad_hidro[2]

    return (sum_centrales_ex + model.falla[bloque])*(1/(1+perdida)) >= model.param_bloques[bloque]['demanda']*model.param_bloques[bloque]['duracion']
    
def max_gen_ex(model, planta, bloque):
    disponibilidad = 1
    if planta not in t_centrales:
        if bloque == 'bloque_1':
            disponibilidad = dispnibilidad_hidro[0]
        elif bloque == 'bloque_2':
            disponibilidad = dispnibilidad_hidro[1]
        elif bloque == 'bloque_3':
            disponibilidad = dispnibilidad_hidro[2]
        return model.generacion[planta, bloque] <= model.param_centrales[planta]['potencia_neta_mw'] * model.param_bloques[bloque]['duracion']*disponibilidad 
    else:
        return model.generacion[planta, bloque] <= model.param_centrales[planta]['potencia_neta_mw'] * model.param_bloques[bloque]['duracion']*model.param_centrales[planta]['disponibilidad']

def max_gen_nuevas(model, planta, bloque):
    disponibilidad = 1
    if planta not in t_centrales:
        if bloque == 'bloque_1':
            disponibilidad = dispnibilidad_hidro[0]
        elif bloque == 'bloque_2':
            disponibilidad = dispnibilidad_hidro[1]
        elif bloque == 'bloque_3':
            disponibilidad = dispnibilidad_hidro[2]
        return model.generacion_nuevas[planta, bloque] <= model.potencia_in_nuevas[planta] * model.param_bloques[bloque]['duracion']*disponibilidad
    else:
        return model.generacion_nuevas[planta, bloque] <= model.potencia_in_nuevas[planta] * model.param_bloques[bloque]['duracion']*model.param_centrales_nuevas[planta]['disponibilidad']

def max_capacidad_nuevas(model, planta):
    if planta in t_ernc:
        return model.potencia_in_nuevas[planta] <= model.param_centrales_nuevas[planta]['maxima_restriccion_2030_MW']
    else:
        return pyo.Constraint.Skip

# Agregar las restricciones al modelo
model.demanda_constraint = pyo.Constraint(model.BLOQUES, rule=balance_demanda)
model.max_gen_constraint = pyo.Constraint(model.CENTRALES, model.BLOQUES, rule=max_gen_ex)
model.max_gen_nuevas_constraint = pyo.Constraint(model.CENTRALES_NUEVAS, model.BLOQUES, rule=max_gen_nuevas)
model.max_capacidad_nuevas_constraint = pyo.Constraint(model.CENTRALES_NUEVAS, rule=max_capacidad_nuevas)

# Función objetivo
model.obj = pyo.Objective(expr=sum(model.generacion[planta, bloque] *(model.param_centrales[planta]['costo_variable_nc']+model.param_centrales[planta]['costo_variable_t']) 
                                  for planta in model.CENTRALES for bloque in model.BLOQUES) +
                           sum(model.falla[bloque] * costo_falla
                                  for bloque in model.BLOQUES), sense=pyo.minimize)


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
