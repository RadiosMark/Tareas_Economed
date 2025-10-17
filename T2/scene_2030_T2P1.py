#%%
# Imports
import pandas as pd
import highspy as hgs
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Parámetros de configuración
tolerancia = 0.00001
perdida = 0.04


#%%
# Cargar archivos CSV (asegúrate que los archivos existan en el mismo directorio)
df_centrales_ex = pd.read_csv('centrales_ex_ED_CS.csv')
df_centrales_ex = df_centrales_ex.fillna(0)

df_centrales_nuevas = pd.read_csv('centrales_n_ED_CS.csv')
df_centrales_nuevas = df_centrales_nuevas.fillna(0)

# Preparar parámetros (index por nombre de planta)
param_centrales = df_centrales_ex.set_index(['planta_n'])
param_centrales_nuevas = df_centrales_nuevas.set_index(['planta_n'])


normas_emision = {'carbon': {'MP': 0.99 , 'SOx' : 0.95, 'NOx' : 0.0},
               'cc-gnl': {'MP': 0.95 , 'SOx' : 0.0, 'NOx' : 0.9}, 
               'petroleo_diesel': {'MP': 0.95 , 'SOx' : 0.0, 'NOx' : 0.0}}
# Revisar las primeras filas
#display(df_centrales_ex.head())
#display(df_centrales_nuevas.head())


#%%

# Cargar el DataFrame
df_combos = pd.read_csv('combos.csv')

# Reemplazar NaN con None
df_combos = df_combos.where(pd.notnull(df_combos), None)

# Convertir el DataFrame a una lista de listas
lista_combos = df_combos.values.tolist()

index_combos = []
for i in range(1, 65):
    index_combos.append(f'combo_{i}')

# Crear un diccionario para los combos: son comunes para centrales existentes y nuevas
dic_combos = {f'combo_{i+1}':lista_combos[i] for i in range(len(lista_combos))}

print(dic_combos)

#%% 
# Equipos de abatimiento
df_equipo_mp = pd.read_csv('abatimiento/equipo_mp.csv')
df_equipo_mp = df_equipo_mp.fillna(0)
equipo_mp = df_equipo_mp.set_index(['Equipo_MP'])
print(equipo_mp)

df_equipo_nox = pd.read_csv('abatimiento/equipo_nox.csv')
df_equipo_nox = df_equipo_nox.fillna(0)
equipo_nox = df_equipo_nox.set_index(['Equipo_NOx'])

df_equipo_sox = pd.read_csv('abatimiento/equipo_sox.csv')
df_equipo_sox = df_equipo_sox.fillna(0)
equipo_sox = df_equipo_sox.set_index(['Equipo_SOx'])

# Daño ambiental (costo social)

df_dano_ambiental = pd.read_csv('dano_ambiental.csv')
df_dano_ambiental = df_dano_ambiental.fillna(0)
dano_ambiental = df_dano_ambiental.set_index(['Ubicacion'])
#%%

dic_equipo = {'MP':equipo_mp.to_dict(orient='index'), 
              'NOx':equipo_nox.to_dict(orient='index'), 
              'SOx':equipo_sox.to_dict(orient='index')}
print(dic_equipo)


#%%

## 
for combo in dic_combos:
    if dic_combos[combo][0] != None:
        tipo = dic_combos[combo][0]
        dic_combos[combo][0] = [tipo, {tipo : dic_equipo['MP'][tipo]}]
    if dic_combos[combo][1] != None:
        tipo = dic_combos[combo][1]
        dic_combos[combo][1] = [tipo, {tipo : dic_equipo['SOx'][tipo]}]
    if dic_combos[combo][2] != None:
        tipo = dic_combos[combo][2]
        dic_combos[combo][2] = [tipo, {tipo : dic_equipo['NOx'][tipo]}]
        
print(dic_combos)



#%%
# Tipos de Centrales
t_centrales = ['biomasa', 'carbon','cc-gnl', 'petroleo_diesel', 'hidro', 'minihidro','eolica','solar', 'geotermia']
t_ernc = ['eolica','solar', 'geotermia','minihidro','hidro' ]
dispnibilidad_hidro = [0.8215,0.6297,0.561]
costo_falla = 505.5
year = 2030 - 2016

# Constantes para pasar de GWh a Toneladas de combustible (calculado en el excel)
conversion_GWh_a_ton = {'petroleo_diesel': 79.6808152,
                         'cc-gnl' : 61.96031117,
                            'carbon': 132.3056959}

## Se supone que si cada uno de estos factores lo multiplicamos GWh * factor * emisiones descontroladas
## nos da las emisiones en kg de cada contaminante (MP, SOx, NOx, CO2) 
## luego hay que pasarlo a toneladas dividiendo por 1000
## y finalmente para pasarlo a costo social hay que multiplicarlo por el costo social de cada contaminante




#%%

# Índices Centrales
index_plantas = param_centrales.index.tolist()
index_plantas_nuevas = param_centrales_nuevas.index.tolist()  # demanda en MW (en la pasada tenia 12k en vez de 1200 xd)

# Índices Equipos de Abatimiento
index_equipo_mp = equipo_mp.index.tolist() # MP-1, MP-2, MP-3
index_equipo_nox = equipo_nox.index.tolist() # NOx-1, NOx-2, NOx-3
index_equipo_sox = equipo_sox.index.tolist() # SOx-1, SOx-2, SOx-3

dic_bloques = {'bloque_1': {'duracion': 1200 , 'demanda' : 10233.87729},
               'bloque_2': {'duracion': 4152 , 'demanda' : 7872.0103}, 
               'bloque_3': {'duracion': 3408 , 'demanda' : 6297.872136}}

normas_emision = {'carbon': {'MP': 0.99 , 'SOx' : 0.95, 'NOx' : 0.0},
               'cc-gnl': {'MP': 0.95 , 'SOx' : 0.0, 'NOx' : 0.9}, 
               'petroleo_diesel': {'MP': 0.95 , 'SOx' : 0.0, 'NOx' : 0.0}}



vida_util = 30 # todos los equipos tienen vida util de 30 años 

# Mostrar resumen
print('Plantillas existentes:', index_plantas[:5])
print('Plantillas nuevas:', index_plantas_nuevas[:5])

#%% 

dic_equipo = {'MP':equipo_mp.to_dict(orient='index'), 
              'NOx':equipo_nox.to_dict(orient='index'), 
              'SOx':equipo_sox.to_dict(orient='index')}
print(dic_equipo)

#print escalonado para mostrar  lo que hay en cada diccionario
for key in dic_equipo:
    print(f"Equipos para {key}:")
    for equipo, valores in dic_equipo[key].items():
        print(f"  {equipo}: {valores}")
        
#%%

# Construcción del modelo Pyomo
model = pyo.ConcreteModel()

# Conjuntos
model.CENTRALES = pyo.Set(initialize=index_plantas)
model.CENTRALES_NUEVAS = pyo.Set(initialize=index_plantas_nuevas)
model.BLOQUES = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])
# creamos el conjunto de 64 combos 
model.COMBOS_EX = pyo.Set(initialize=index_combos)  # Conjunto de plantas existentes
model.COMBOS_NUEVAS = pyo.Set(initialize=index_combos)  # Conjunto de plantas nuevas


# Parámetros en formato dict (Pyomo Any para permitir dicts anidados)
model.param_centrales = pyo.Param(model.CENTRALES, initialize=param_centrales.to_dict(orient='index'), within=pyo.Any)
model.param_centrales_nuevas = pyo.Param(model.CENTRALES_NUEVAS, initialize=param_centrales_nuevas.to_dict(orient='index'), within=pyo.Any)
model.param_bloques = pyo.Param(model.BLOQUES, initialize=dic_bloques, within=pyo.Any)
model.param_combos_ex = pyo.Param(model.COMBOS_EX, initialize=dic_combos, within=pyo.Any)
model.param_combos_nuevas = pyo.Param(model.COMBOS_NUEVAS, initialize=dic_combos, within=pyo.Any)

# Variables(todas las generaciones son en GWh y la potencia en MW)
model.generacion_ex = pyo.Var(model.CENTRALES, model.BLOQUES, within=pyo.NonNegativeReals)
model.generacion_nuevas = pyo.Var(model.CENTRALES_NUEVAS, model.BLOQUES, within=pyo.NonNegativeReals)
model.potencia_in_nuevas = pyo.Var(model.CENTRALES_NUEVAS, within=pyo.NonNegativeReals)
model.falla = pyo.Var(model.BLOQUES, within=pyo.NonNegativeReals)

model.eleccion_combo_ex = pyo.Var(model.CENTRALES, model.COMBOS_EX, within=pyo.Binary)
model.eleccion_combo_nuevas = pyo.Var(model.CENTRALES_NUEVAS, model.COMBOS_NUEVAS, within=pyo.Binary)

# Variable binaria para habilitar/deshabilitar la operación de las centrales
model.operacion_ex = pyo.Var(model.CENTRALES, within=pyo.Binary)
model.operacion_new = pyo.Var(model.CENTRALES_NUEVAS, within=pyo.Binary)

# %%

# Restricciones
def fd_hidro(bloque):
    if bloque == 'bloque_1':
        return dispnibilidad_hidro[0]
    elif bloque == 'bloque_2':
        return dispnibilidad_hidro[1]
    else:
        return dispnibilidad_hidro[2]

def balance_demanda(model, bloque):
    gen_ex = sum(model.generacion_ex[planta, bloque] for planta in model.CENTRALES)
    gen_new = sum(model.generacion_nuevas[planta, bloque] for planta in model.CENTRALES_NUEVAS)
    return (gen_ex + gen_new + model.falla[bloque]) * (1000/(1+perdida)) >= model.param_bloques[bloque]['demanda'] * model.param_bloques[bloque]['duracion']
                                                    # el 1000 es para convertir de GWh a MWh

def max_gen_ex(model, planta, bloque):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['hidro', 'minihidro']:
        disp = fd_hidro(bloque)
    else:
        disp = model.param_centrales[planta]['disponibilidad']
    # generacion está en GWh (multiplicamos por 1000  para hacer el cambio a MWh), potencia_neta_mw en MW, duracion en horas
    return model.generacion_ex[planta, bloque] * 1000 <= model.operacion_ex[planta] * model.param_centrales[planta]['potencia_neta_mw'] * model.param_bloques[bloque]['duracion'] * disp

def max_gen_nuevas(model, planta, bloque):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['hidro', 'minihidro']:
        disp = fd_hidro(bloque)
    else:
        disp = model.param_centrales_nuevas[planta]['disponibilidad']
    # generacion está en GWh (multiplicamos por 1000  para hacer el cambio a MWh), potencia_neta_mw en MW, duracion en horas
    return model.generacion_nuevas[planta, bloque] * 1000 <= model.operacion_new[planta] * model.potencia_in_nuevas[planta] * model.param_bloques[bloque]['duracion'] * disp

def max_capacidad_nuevas(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in t_ernc:
        limite = model.param_centrales_nuevas[planta]['maxima_restriccion_2030_MW']
        return model.potencia_in_nuevas[planta] <= limite if limite > 0 else pyo.Constraint.Skip # Si el límite es 0, no hay restricción
    else:
        return pyo.Constraint.Skip
    
def anualidad(r, n): # r es tasa de descuento, n es vida util en años
    return r / (1 - (1 + r)**(-n))


# Restricción para limitar la selección de combos
def restriccion_combos_ex(model, planta):
    tecnologia = model.param_centrales[planta]['tecnologia']
    if tecnologia in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        # Permitir que solo se seleccione un combo entre todos
        return sum(model.eleccion_combo_ex[planta, combo] for combo in model.COMBOS_EX) == 1
    else:
        # Restringir a que solo se seleccione el primer combo
        return model.eleccion_combo_ex[planta, 'combo_1'] == 1
    
def restriccion_combos_new(model, planta):
    tecnologia = model.param_centrales_nuevas[planta]['tecnologia']
    if tecnologia in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        # Permitir que solo se seleccione un combo entre todos
        return sum(model.eleccion_combo_nuevas[planta, combo] for combo in model.COMBOS_NUEVAS) == 1
    else:
        # Restringir a que solo se seleccione el primer combo
        return model.eleccion_combo_nuevas[planta, 'combo_1'] == 1




# Definir una expresión para calcular la eficiencia ponderada para existentes y nuevas
def calcular_efi_mp_ex(model, planta):
    return sum(
        model.eleccion_combo_ex[planta, combo] * dic_equipo['MP'][dic_combos[combo][0]]['Eficiencia_(p.u.)']
        for combo in model.COMBOS_EX if dic_combos[combo][0] is not None
    )

def calcular_efi_nox_ex(model, planta):
    return sum(
        model.eleccion_combo_ex[planta, combo] * dic_equipo['SOx'][dic_combos[combo][1]]['Eficiencia_(p.u.)']
        for combo in model.COMBOS_EX if dic_combos[combo][1] is not None
    )

def calcular_efi_sox_ex(model, planta):
    return sum(
        model.eleccion_combo_ex[planta, combo] * dic_equipo['NOx'][dic_combos[combo][2]]['Eficiencia_(p.u.)']
        for combo in model.COMBOS_EX if dic_combos[combo][2] is not None
    )

def calcular_efi_mp_new(model, planta):
    return sum(
        model.eleccion_combo_nuevas[planta, combo] * dic_equipo['MP'][dic_combos[combo][0]]['Eficiencia_(p.u.)']
        for combo in model.COMBOS_NUEVAS if dic_combos[combo][0] is not None
    )

def calcular_efi_nox_new(model, planta):
    return sum(
        model.eleccion_combo_nuevas[planta, combo] * dic_equipo['SOx'][dic_combos[combo][1]]['Eficiencia_(p.u.)']
        for combo in model.COMBOS_NUEVAS if dic_combos[combo][1] is not None
    )

def calcular_efi_sox_new(model, planta):
    return sum(
        model.eleccion_combo_nuevas[planta, combo] * dic_equipo['NOx'][dic_combos[combo][2]]['Eficiencia_(p.u.)']
        for combo in model.COMBOS_NUEVAS if dic_combos[combo][2] is not None
    )


# Agregar la expresión al modelo
model.efi_mp = pyo.Expression(model.CENTRALES, rule=calcular_efi_mp_ex)
model.efi_nox = pyo.Expression(model.CENTRALES, rule=calcular_efi_nox_ex)
model.efi_sox = pyo.Expression(model.CENTRALES, rule=calcular_efi_sox_ex)

model.efi_mp_new = pyo.Expression(model.CENTRALES_NUEVAS, rule=calcular_efi_mp_new)
model.efi_nox_new = pyo.Expression(model.CENTRALES_NUEVAS, rule=calcular_efi_nox_new)
model.efi_sox_new = pyo.Expression(model.CENTRALES_NUEVAS, rule=calcular_efi_sox_new)

# Restricción para garantizar que la eficiencia cumpla con las normas
def restriccion_emi_mp_ex(model, planta):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.efi_mp[planta] >= normas_emision[tec]['MP'] 
    else:
        return pyo.Constraint.Skip

def restriccion_emi_nox_ex(model, planta):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.efi_nox[planta] >= normas_emision[tec]['NOx'] 
    else:
        return pyo.Constraint.Skip
    
def restriccion_emi_sox_ex(model, planta):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.efi_sox[planta] >= normas_emision[tec]['SOx'] 
    else:
        return pyo.Constraint.Skip

def restriccion_emi_mp_new(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.efi_mp_new[planta] >= normas_emision[tec]['MP'] 
    else:
        return pyo.Constraint.Skip
    
def restriccion_emi_nox_new(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.efi_nox_new[planta] >= normas_emision[tec]['NOx'] 
    else:
        return pyo.Constraint.Skip
    
def restriccion_emi_sox_new(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.efi_sox_new[planta] >= normas_emision[tec]['SOx'] 
    else:
        return pyo.Constraint.Skip

def restriccion_normas_operacion_mp_ex(model, planta):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.operacion_ex[planta] <= (model.efi_mp[planta] >= normas_emision[tec]['MP'])
    else:
        return pyo.Constraint.Skip

def restriccion_normas_operacion_mp_new(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.operacion_new[planta] <= (model.efi_mp_new[planta] >= normas_emision[tec]['MP'])
    else:
        return pyo.Constraint.Skip

def restriccion_normas_operacion_nox_new(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.operacion_new[planta] <= (model.efi_nox_new[planta] >= normas_emision[tec]['NOx'])
    else:
        return pyo.Constraint.Skip

def restriccion_normas_operacion_sox_new(model, planta):
    tec = model.param_centrales_nuevas[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.operacion_new[planta] <= (model.efi_sox_new[planta] >= normas_emision[tec]['SOx'])
    else:
        return pyo.Constraint.Skip


def restriccion_normas_operacion_nox_ex(model, planta):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.operacion_ex[planta] <= (model.efi_nox[planta] >= normas_emision[tec]['NOx'])
    else:
        return pyo.Constraint.Skip

def restriccion_normas_operacion_sox_ex(model, planta):
    tec = model.param_centrales[planta]['tecnologia']
    if tec in ['petroleo_diesel', 'carbon', 'cc-gnl']:
        return model.operacion_ex[planta] <= (model.efi_sox[planta] >= normas_emision[tec]['SOx'])
    else:
        return pyo.Constraint.Skip

# Agregar la restricción al modelo
model.restriccion_emisiones_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_emi_mp_ex)
model.restriccion_emisiones_nox_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_emi_nox_ex)
model.restriccion_emisiones_sox_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_emi_sox_ex)

model.restriccion_emisiones_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_emi_mp_new)
model.restriccion_emisiones_nox_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_emi_nox_new)
model.restriccion_emisiones_sox_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_emi_sox_new)

# Restricciones para plantas existentes
model.restriccion_normas_operacion_mp_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_normas_operacion_mp_ex)
model.restriccion_normas_operacion_nox_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_normas_operacion_nox_ex)
model.restriccion_normas_operacion_sox_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_normas_operacion_sox_ex)

# Restricciones para plantas nuevas
model.restriccion_normas_operacion_mp_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_normas_operacion_mp_new)
model.restriccion_normas_operacion_nox_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_normas_operacion_nox_new)
model.restriccion_normas_operacion_sox_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_normas_operacion_sox_new)

# Adjuntar restricciones

model.max_gen_constraint = pyo.Constraint(model.CENTRALES, model.BLOQUES, rule=max_gen_ex)
model.max_gen_nuevas_constraint = pyo.Constraint(model.CENTRALES_NUEVAS, model.BLOQUES, rule=max_gen_nuevas)
model.max_capacidad_nuevas_constraint = pyo.Constraint(model.CENTRALES_NUEVAS, rule=max_capacidad_nuevas)

# nuevas restricciones
model.restriccion_combos_ex = pyo.Constraint(model.CENTRALES, rule=restriccion_combos_ex)
model.restriccion_combos_new = pyo.Constraint(model.CENTRALES_NUEVAS, rule=restriccion_combos_new)

# Función objetivo (expresiones)

def costo_mp_ex(model,planta):
    return  sum(model.eleccion_combo_ex[planta, combo] * (model.param_combos_ex[combo][0][1][model.param_combos_ex[combo][0][0]]['Costo_variable_($/MWh)'] if model.param_combos_ex[combo][0] is not None else 0) 
                                                         for combo in model.COMBOS_EX)

model.costo_mp_ex = pyo.Expression(model.CENTRALES, rule=costo_mp_ex)

model.costo_abatimiento_ex = pyo.Expression(expr=sum(model.generacion_ex[planta, bloque] * 1000 * model.costo_mp_ex[planta]
                                                    for planta in model.CENTRALES
                                                    for bloque in model.BLOQUES))



# Costo operación (costos variables) 2030
model.op_ex = pyo.Expression(expr=sum(model.generacion_ex[planta, bloque] * 1000 * # pasamos a MWh
                                (model.param_centrales[planta]['costo_variable_nc']
                                + model.param_centrales[planta]['costo_variable_t']
                                + model.param_centrales[planta]['costo_var_comb_16_usd_mwh']) 
                                for planta in model.CENTRALES
                                for bloque in model.BLOQUES)) # esto queda en USD

model.op_new = pyo.Expression(expr=sum(model.generacion_nuevas[planta, bloque] * 1000 * # pasamos a MWh
                                (model.param_centrales_nuevas[planta]['cvnc_usd_MWh'] 
                                + model.param_centrales_nuevas[planta]['linea_peaje_usd_MWh']
                                + model.param_centrales_nuevas[planta]['costo_var_comb_16_usd_mwh'])  
                                for planta in model.CENTRALES_NUEVAS 
                                for bloque in model.BLOQUES)) # esto queda en USD

# Costo inversión (anualidad) 2030
model.inv_new = pyo.Expression(expr=sum(model.potencia_in_nuevas[planta]* 
                                        anualidad(model.param_centrales_nuevas[planta]['tasa_descuento'], model.param_centrales_nuevas[planta]['vida_util_anos']) * 
                                        model.param_centrales_nuevas[planta]['inversion_usd_kW_neto']* 1000 # pasamos a MW
                                        for planta in model.CENTRALES_NUEVAS)) # esto queda en USD/año

# costo FALLAS (recordar que falla está en GWh)
model.costo_fallas = pyo.Expression(expr=sum(model.falla[bloque] * 1000 * # pasamos a MWh
                            costo_falla for bloque in model.BLOQUES))


# costos de operacion de abatidores



model.costo_abatimiento_ex = pyo.Expression(expr=sum(model.generacion_ex[planta, bloque] * 1000 *
                                                     sum(model.eleccion_combo_ex[planta, combo] * 
                                                         (model.param_combos_ex[combo][0][1][model.param_combos_ex[combo][0][0]]['Costo_variable_($/MWh)'] if model.param_combos_ex[combo][0] is not None else 0) 
                                                         for combo in model.COMBOS_EX)
                                                          # pasamos a MWh
                                                     for planta in model.CENTRALES
                                for bloque in model.BLOQUES))


r_df = 0.01
df_2016_2030 = 1 / (1 + r_df)**(year)

# minimizar (costo operacion + inversion + costo falla)
model.obj = pyo.Objective(expr = df_2016_2030 * (model.op_ex + model.op_new + model.inv_new + model.costo_fallas), sense = pyo.minimize)

# %%
