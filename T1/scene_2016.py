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

#%%

param_centrales = df_centrales_ex.set_index(['planta_n'])

#%%

# Tipos de Centrales
t_centrales = ['biomasa', 'carbon','cc-gnl', 'petroleo_diesel', 'eolica','solar', 'geotermia']

# Para la modelación de la hidroelectricidad
dispnibilidad_hidro = [0.8215,0.6297,0.561]

costo_falla = 505.5 # mills/kWh = USD/MWh

#%%

index_plantas = param_centrales.index.tolist()
dic_bloques = {'bloque_1': {'duracion': 12000 , 'demanda' : 7756},
               'bloque_2': {'duracion': 4152 , 'demanda' : 5966}, 
               'bloque_3': {'duracion': 3408 , 'demanda' : 4773}}

#%%

# MODELO
model = pyo.ConcreteModel()

# CONJUNTOS

model.CENTRALES = pyo.Set(initialize=index_plantas)
model.BLOQUES = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])

# PARAMETROS
model.param_centrales = pyo.Param(model.CENTRALES,
                                  initialize=param_centrales.to_dict(orient='index'),
                                  within=pyo.Any)

model.param_bloques = pyo.Param(model.BLOQUES, initialize=dic_bloques, within=pyo.Any)

# VARIABLES

model.generacion = pyo.Var(model.CENTRALES, model.BLOQUES, within=pyo.NonNegativeReals) 
model.falla = pyo.Var(model.BLOQUES, within=pyo.NonNegativeReals)

# RESTRICCIONES

def fd_hidro(bloque):
    return dispnibilidad_hidro[0] if bloque == 'bloque_1' else \
           dispnibilidad_hidro[1] if bloque == 'bloque_2' else \
           dispnibilidad_hidro[2]

def balance_demanda(model, bloque):
    # Energía neta generada (todas las tecnologías)
    suma = sum(model.generacion[planta, bloque] for planta in model.CENTRALES)

    # Opción B (si quieres aplicar pérdidas también a la falla):
    return (suma + model.falla[bloque])*(1/(1+perdida)) \
                        >= model.param_bloques[bloque]['demanda'] * model.param_bloques[bloque]['duracion']

def max_gen(model, planta, bloque):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['hidro', 'hidro_conv', 'minihidro']:
        disp = fd_hidro(bloque)
    else:
        # cuidado: si del CSV vino 0 por fillna, usa 1.0 como fallback
        disp = model.param_centrales[planta]['disponibilidad'] or 1.0
        #disp = disp_val if disp_val not in (None, 0) else 1.0  # fallback solo si está vacío/0

    return model.generacion[planta, bloque] \
           <= model.param_centrales[planta]['potencia_neta_mw'] \
              * model.param_bloques[bloque]['duracion'] * disp


# Restricciones al modelo
model.demanda_constraint = pyo.Constraint(model.BLOQUES, rule=balance_demanda)
model.max_gen_constraint = pyo.Constraint(model.CENTRALES, model.BLOQUES, rule=max_gen)


# operación EXISTENTES
model.costo_op_ex = pyo.Expression(
    expr=sum(
        model.generacion[planta, bloque] *
        (model.param_centrales[planta]['costo_variable_nc'] +
         model.param_centrales[planta]['costo_variable_t'])
        for planta in model.CENTRALES 
        for bloque in model.BLOQUES
    )
)

# costo FALLAS
model.costo_fallas = pyo.Expression(
    expr=sum(model.falla[bloque] * costo_falla for bloque in model.BLOQUES)
)

# FUNCION OBJETIVO FINAL
model.obj = pyo.Objective(
    expr = model.costo_op_ex + model.costo_fallas,
    sense = pyo.minimize
)


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
