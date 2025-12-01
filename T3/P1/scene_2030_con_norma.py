# %% 
import pandas as pd
import re
import highspy as hgs
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from collections import defaultdict
import math

# Parámetros de configuración
tolerancia = 0.00001
perdida = 0.04

# Tipos de Centrales
t_centrales = ['biomasa', 'carbon','cc-gnl', 'petroleo_diesel', 'hidro', 'minihidro','eolica','solar', 'geotermia']
t_ernc = ['eolica','solar', 'geotermia','minihidro','hidro' ]

# diccionearios de abatidores
# Equipos de abatimiento
df_equipo_mp = pd.read_csv('abatidores/equipo_mp.csv')
df_equipo_mp = df_equipo_mp.fillna(0)
equipo_mp = df_equipo_mp.set_index(['Equipo_MP'])

df_equipo_nox = pd.read_csv('abatidores/equipo_nox.csv')
df_equipo_nox = df_equipo_nox.fillna(0)
equipo_nox = df_equipo_nox.set_index(['Equipo_NOx'])

df_equipo_sox = pd.read_csv('abatidores/equipo_sox.csv')
df_equipo_sox = df_equipo_sox.fillna(0)
equipo_sox = df_equipo_sox.set_index(['Equipo_SOx'])

dic_equipo = {'MP':equipo_mp.to_dict(orient='index'), 
              'NOx':equipo_nox.to_dict(orient='index'), 
              'SOx':equipo_sox.to_dict(orient='index')}



dispnibilidad_hidro = [0.8215,0.6297,0.561]
costo_falla = 500  # $/MWh
year = 2030 - 2016

def excel_range_to_df(path, sheet='existentes', cell_range='E7:AB2695', header=None, engine='openpyxl'):
    """
    Leer un rango específico de un Excel usando pandas.
    - path: Ruta al archivo Excel.
    - sheet: Nombre de la hoja.
    - cell_range: Rango en formato 'E7:AB2695'.
    - header: None (sin cabecera) o int para indicar la fila de cabecera relativa al rango.
    """
    m = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', cell_range.upper())
    if not m:
        raise ValueError("Rango debe tener formato 'E7:AB2695'")
    c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    usecols = f"{c1}:{c2}"
    skiprows = r1 - 1
    nrows = r2 - r1 + 1
    return pd.read_excel(path, sheet_name=sheet, usecols=usecols,
                         skiprows=skiprows, nrows=nrows, header=header, engine=engine)

def obtener_tecnologia(central):
    # central es del tipo "petroleo_diesel|diego_de_almagro"
    if central == 'central_falla':
        return 'central_falla'
    else:
        index = central.find('|')
        return central[:index].strip()

def fd_hidro(bloque):
    if bloque == 'bloque_1':
        return dispnibilidad_hidro[0]
    elif bloque == 'bloque_2':
        return dispnibilidad_hidro[1]
    else:
        return dispnibilidad_hidro[2]

def anualidad(r, n): # r es tasa de descuento, n es vida util en años
    return r / (1 - (1 + r)**(-n))

# Leer el archivo 'datos_t2.xlsx', hoja 'existentes', rango 'E7:X2695'
excel_path = 'datos_t2.xlsx'
df_existentes = excel_range_to_df(excel_path, sheet='existentes', cell_range='E7:AB2695', header=0)



CONJ = df_existentes.set_index('id_combinacion').to_dict(orient='index')
CONJ_C = df_existentes['id_centralcomb'].unique().tolist()

HEADERS = df_existentes.columns.tolist()
""" 
Obtengo esto
['id_combinacion',
 'id_centralcomb',
 'Central',
 'MP',
 'Sox',
 'Nox',
 'costo_fijo($/KW-neto)',
 'costo variable($/MWh)',
 'potencia_neta(MW)',
 'eficiencia(%)',
 'disponibilidad(p.u.)',
 'vida_util',
 'Restricciones_max(MW)',
 'ED_MP(kg/Mg)',
 'ED_Nox(kg/Mg)',
 'ED_Sox(kg/Mg)',
 'ED_CO2(kg/Mg)',
 'Norma_MP',
 'Norma_Sox',
 'Norma_Nox',
 'CS_MP($/ton)',
 'CS_Sox($/ton)',
 'CS_Nox($/ton)',
 'CS_Co2($/ton)'] """

# %%

# --- 1. CARGA DE DATOS (Tu código actual) ---
CONJ = df_existentes.set_index('id_combinacion').to_dict(orient='index')
CONJ_C = df_existentes['id_centralcomb'].unique().tolist()


# --- 2. PREPARACIÓN PARA PYOMO ---

# Crear el mapeo de Central -> [Combinaciones]
MAPEO_C_a_I = defaultdict(list)
for i, data in CONJ.items():
    c = data['id_centralcomb']
    MAPEO_C_a_I[c].append(i)
MAPEO_C_a_I = dict(MAPEO_C_a_I) # Convertir a dict normal


dic_bloques = {'bloque_1': {'duracion': 1200 , 'demanda' : 10233.87729},
               'bloque_2': {'duracion': 4152 , 'demanda' : 7872.0103}, 
               'bloque_3': {'duracion': 3408 , 'demanda' : 6297.872136}}


central_falla = 225 # posición de la central de falla
tasa_descuento = 0.1

# --- 3. CONSTRUCCIÓN DEL MODELO PYOMO ---

## nocion: ahora quie tenemos 64 combinaciones para todas las centrales, 
## el indice por central i = 64(c−1)+a 
## c = central y a es la combinacion dentro de la central
## c in {1,2,...,num_centrales}
## a in {1,2,...,num_combinaciones_por_central}

model = pyo.ConcreteModel(name="Modelo_Base")

# --- CONJUNTOS ---
model.I = pyo.Set(initialize=CONJ.keys()) # Combinaciones globales 2688
model.C = pyo.Set(initialize=CONJ_C)      # Centrales: 42
model.B = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])

# --- PARÁMETROS ---
model.tecnologia = pyo.Param(model.I, initialize=lambda m, i: obtener_tecnologia(CONJ[i]['Central']))
model.potencia_neta = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['potencia_neta(MW)'])
model.potencia_max = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['Restricciones_max(MW)'])
model.disponibilidad = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['disponibilidad(p.u.)'])
model.eficiencia = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['eficiencia(%)'])
model.costo_fijo = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['costo_fijo($/KW-neto)'])
model.costo_variable = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['costo variable($/MWh)'])
model.abatidor_mp = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['MP']) 
model.abatidor_sox = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['Sox']) 
model.abatidor_nox = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['Nox']) 
model.vida_util = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['vida_util'])
model.param_bloques = pyo.Param(model.B, initialize=dic_bloques)
# Costos sociales
model.costo_social_mp = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_MP($/ton)'])
model.costo_social_sox = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_Sox($/ton)'])
model.costo_social_nox = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_Nox($/ton)'])
model.costo_social_co2 = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_Co2($/ton)'])

# Parámetro Mutable para el Límite de CO2 (se actualizará en el loop)
# Se inicializa en infinito para la primera corrida (BAU)
model.limite_co2_total = pyo.Param(mutable=True, initialize=float('inf'))

# --- CONVERSIÓN DE UNIDADES Y CÁLCULO DE NORMA ---

# 1. Poder Calorífico (calculado desde tu tabla "Caja Mágica")
poder_calorifico = {
    'carbon':          1 / 132.3056959,
    'petroleo_diesel': 1 / 79.6808152,
    'cc-gnl':          1 / 61.96031117
}

# 2. Función para convertir unidades de [kg/ton] a [ton/GWh]
def convertir_unidades(tec, valor_kg_ton):
    cal_val = poder_calorifico.get(tec)
    if cal_val is None or math.isnan(valor_kg_ton):
        return 0
    return (valor_kg_ton / 1000) / cal_val

# 3. Diccionario para guardar la emisión "descontrolada" de cada central
emision_descontrolada = defaultdict(dict)
for c in model.C:
    primer_i = MAPEO_C_a_I[c][0] 
    tec = model.tecnologia[primer_i]
    emision_descontrolada[c]['NOx'] = convertir_unidades(tec, CONJ[primer_i]['ED_Nox(kg/Mg)'])
    emision_descontrolada[c]['SOx'] = convertir_unidades(tec, CONJ[primer_i]['ED_Sox(kg/Mg)'])
    emision_descontrolada[c]['MP']  = convertir_unidades(tec, CONJ[primer_i]['ED_MP(kg/Mg)'])
    emision_descontrolada[c]['CO2'] = convertir_unidades(tec, CONJ[primer_i]['ED_CO2(kg/Mg)']) 


# 4. Funciones para cargar los parámetros (versión corregida)
def cargar_factor_emision(m, i, cont_corto, cont_largo):
    tec = m.tecnologia[i]
    valor_kg_ton = CONJ[i][f'ED_{cont_largo}(kg/Mg)']
    return convertir_unidades(tec, valor_kg_ton)

# calculamos el limite de emision absoluta segun norma
def cargar_norma_absoluta(m, i, cont_corto, cont_largo):
    id_central = CONJ[i]['id_centralcomb']
    ed_base = emision_descontrolada[id_central][cont_corto]
    porcentaje_reduccion = CONJ[i][f'Norma_{cont_largo}']
    
    if math.isnan(porcentaje_reduccion):
        return float('inf') # Si no hay norma, el límite es infinito
        
    return ed_base * (1 - porcentaje_reduccion)

# 5. Carga final de parámetros pasa de gwh a ton de contaminante
model.factor_emision_Nox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'NOx', 'Nox'))
model.factor_emision_Sox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'SOx', 'Sox'))
model.modelo_emision_MP = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'MP', 'MP'))
model.factor_emision_CO2 = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'CO2', 'CO2'))

# B. Carga de normas de emisión absoluta, cuanto es lo máximo permitido de emitir
model.norma_emision_Nox = pyo.Param(model.I, initialize=lambda m, i: cargar_norma_absoluta(m, i, 'NOx', 'Nox'))
model.norma_emision_Sox = pyo.Param(model.I, initialize=lambda m, i: cargar_norma_absoluta(m, i, 'SOx', 'Sox'))
model.norma_emision_MP = pyo.Param(model.I, initialize=lambda m, i: cargar_norma_absoluta(m, i, 'MP', 'MP'))

# --- VARIABLES ---
model.P = pyo.Var(model.I, within=pyo.NonNegativeReals)            # potencia instalada [MW] por combinacion
model.E = pyo.Var(model.I, model.B, within=pyo.NonNegativeReals)  # energia generada [GWh] en el bloque


# %%

# Restricción 1: Balance de demanda por bloque
def balance_demanda(m,b):
    generacion_total = sum(m.E[i,b] for i in m.I)
    return generacion_total* (1000/(1+perdida)) >= \
            m.param_bloques[b]['demanda']* m.param_bloques[b]['duracion']

# Restriccion 2: Se mantienen la potencia las centrales existentes
def potencia_existente(m, c):
    combinaciones_de_c = MAPEO_C_a_I[c] # equivalente a i = 64 * (c - 1) + 1 
    primer_i = combinaciones_de_c[0]
    pot_neta = m.potencia_neta[primer_i]
    if math.isnan(pot_neta): # nuevas
        return pyo.Constraint.Skip
    else:
        return sum(m.P[i] for i in combinaciones_de_c) == pot_neta

# Restriccion 3: Dispobibilidad Técnica (maxima generacion)
""" def disponibilidad_tecnica(m, c, b):
    combinaciones_de_c = MAPEO_C_a_I[c] # equivalente a i = 64 * (c - 1) + 1 
    primer_i = combinaciones_de_c[0] 

    if m.tecnologia[primer_i] == 'central_falla':
        return pyo.Constraint.Skip

    if m.tecnologia[primer_i] in ['hidro', 'hidro_conv', 'minihidro']: 
        disp = fd_hidro(b)
    else:
        disp = m.disponibilidad[primer_i]

    generacion_central = sum(m.E[i, b] for i in combinaciones_de_c) #GWh
    potencia_instalada_central = sum(m.P[i] for i in combinaciones_de_c) #MW
            # mwh                             
    return (generacion_central * 1000) <= potencia_instalada_central * disp * m.param_bloques[b]['duracion']
 """

def disponibilidad_tecnica(m, i, b):
    if m.tecnologia[i] == 'central_falla':
        return pyo.Constraint.Skip

    if m.tecnologia[i] in ['hidro', 'hidro_conv', 'minihidro']: 
        disp = fd_hidro(b)
    else:
        disp = m.disponibilidad[i]
    return (m.E[i, b] * 1000) <= m.P[i] * disp * m.param_bloques[b]['duracion']

# Restriccion 4: Capacidad Por Central (esto seria para las nuevas, impone el techo)
def capacidad_por_central(m, c):
    combinaciones_de_c = MAPEO_C_a_I[c] # equivalente a i = 64 * (c - 1) + 1 
    primer_i = combinaciones_de_c[0]
    pot_max = m.potencia_max[primer_i] # Obtener el valor primero
    if math.isnan(pot_max):
        return pyo.Constraint.Skip
    else:
        return sum(m.P[i] for i in combinaciones_de_c) <= pot_max

# Restrccion 5.1 : Norma de Emisión NOx
def norma_emision_nox(m,i,b):
    tec = m.tecnologia[i]
    if tec in ['petroleo_diesel','carbon','cc-gnl']:
        abatidor = m.abatidor_nox[i]
        
        # Si no hay abatidor, la eficiencia es 0. Si lo hay, la buscamos.
        if not isinstance(abatidor, str):
            efi_aba = 0
        else:
            efi_aba = dic_equipo['NOx'][abatidor]['Eficiencia_(p.u.)']

        norma = m.norma_emision_Nox[i]
        ed = m.factor_emision_Nox[i]
        efi_calor = m.eficiencia[i]
        
        if norma == float('inf') or efi_calor <= 0:
            return pyo.Constraint.Skip
        else:
            energia_combustible = m.E[i,b] / efi_calor
            return energia_combustible * norma >= energia_combustible * ed * (1 - efi_aba)
    else:
        return pyo.Constraint.Skip

# Restrccion 5.2 : Norma de Emisión MP
def norma_emision_sox(m,i,b):
    tec = m.tecnologia[i]
    if tec in ['petroleo_diesel','carbon','cc-gnl']:
        abatidor = m.abatidor_sox[i]

        if not isinstance(abatidor, str):
            efi_aba = 0
        else:
            efi_aba = dic_equipo['SOx'][abatidor]['Eficiencia_(p.u.)']
        
        norma = m.norma_emision_Sox[i]
        ed = m.factor_emision_Sox[i]
        efi_calor = m.eficiencia[i]

        if norma == float('inf') or efi_calor <= 0:
            return pyo.Constraint.Skip 
        else:
            energia_combustible = m.E[i,b] / efi_calor
            return energia_combustible * norma >= energia_combustible * ed * (1 - efi_aba)
    else:
        return pyo.Constraint.Skip

# Restrccion 5.3 : Norma de Emisión MP
def norma_emision_mp(m,i,b):
    tec = m.tecnologia[i]
    if tec in ['petroleo_diesel','carbon','cc-gnl']:
        abatidor = m.abatidor_mp[i]
        if not isinstance(abatidor, str):
            efi_aba = 0
        else:
            efi_aba = dic_equipo['MP'][abatidor]['Eficiencia_(p.u.)']

        norma = m.norma_emision_MP[i]
        ed = m.modelo_emision_MP[i]
        efi_calor = m.eficiencia[i]

        if norma == float('inf') or efi_calor <= 0:
            return pyo.Constraint.Skip 
        else:
            energia_combustible = m.E[i,b] / efi_calor
            return energia_combustible * norma >= energia_combustible * ed * (1 - efi_aba)
    else:
        return pyo.Constraint.Skip

# Restriccion 6: Norma de Emisión CO2

# Agrega esto en la sección de Restricciones
def restriccion_meta_co2(m):
    emisiones_totales = 0
    for i in m.I:
        tec = m.tecnologia[i]
        # Solo cuentan las térmicas con eficiencia definida
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                # Emisión = (Energía Generada / Eficiencia) * Factor Emisión
                energia_combustible = m.E[i, b] / m.eficiencia[i]
                emisiones_totales += energia_combustible * m.factor_emision_CO2[i]
    
    return emisiones_totales <= m.limite_co2_total


model.demanda_constraint = pyo.Constraint(model.B, rule=balance_demanda)
model.potencia_existente_constraint = pyo.Constraint(model.C, rule=potencia_existente)
model.disponibilidad_tecnica_constraint = pyo.Constraint(model.I, model.B, rule=disponibilidad_tecnica)
model.capacidad_por_central_constraint = pyo.Constraint(model.C, rule=capacidad_por_central)
model.norma_emision_nox_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_nox)
model.norma_emision_sox_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_sox)
model.norma_emision_mp_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_mp)
model.meta_co2_constraint = pyo.Constraint(rule=restriccion_meta_co2)


#%%

# --- FUNCIÓN OBJETIVO ---

# Costo Variable
def costo_operacion(m):
    total_variable = 0
    for b in m.B:
        for i in m.I: 
            # Costo de falla
            if m.tecnologia[i] == 'central_falla':
                total_variable += m.E[i, b] *1000* costo_falla ## faltaba un por mil
                continue

            # Costo variable de la combinación
            costo_var = m.costo_variable[i]

            costo_mp = 0
            costo_sox = 0
            costo_nox = 0

            if m.tecnologia[i] in ['carbon', 'petroleo_diesel', 'cc-gnl']:
                abatidor_mp = m.abatidor_mp[i]
                if isinstance(abatidor_mp, str):
                    costo_mp = dic_equipo['MP'][abatidor_mp]['Costo_variable_($/MWh)']

                abatidor_sox = m.abatidor_sox[i]
                if isinstance(abatidor_sox, str):
                    costo_sox = dic_equipo['SOx'][abatidor_sox]['Costo_variable_($/MWh)']

                abatidor_nox = m.abatidor_nox[i]
                if isinstance(abatidor_nox, str):
                    costo_nox = dic_equipo['NOx'][abatidor_nox]['Costo_variable_($/MWh)']
                
                abatidores_cost = (costo_mp + costo_sox + costo_nox) # $/MWh

                # Energía está en GWh y los costos en $/MWh
                total_variable += m.E[i, b] * 1000 * (costo_var + abatidores_cost)
            else:
                total_variable += m.E[i, b] * 1000 * costo_var
    return total_variable
        
# Costo Fijo (VERSIÓN CON FILTRO DE TECNOLOGÍA)
def costo_fijo(m):
    total_fijo_expression = 0.0
    
    # --- Parámetros de Abatidores (según pauta) ---
    vida_util_abatidores = 30 
    tasa_descuento_abatidores = 0.1 
    anualizacion_abatidores = anualidad(tasa_descuento_abatidores, vida_util_abatidores)

    for i in m.I:
        # pasamos mw a kw
        potencia_instalada_kw = m.P[i] * 1000
        
        costo_anual_central_por_kw = 0.0     # en $/kW-año
        costo_anual_abatidores_por_kw = 0.0  # en $/kW-año

        # --- 1. ¿Es una central NUEVA? Calcular su costo de inversión anualizado ---
        if math.isnan(m.potencia_neta[i]): 
            costo_inv_central = m.costo_fijo[i] # ($/kW)
            
            if costo_inv_central > 0 and m.vida_util[i] > 0:
                anualizacion_central = anualidad(tasa_descuento, m.vida_util[i])
                costo_anual_central_por_kw = costo_inv_central * anualizacion_central

        # --- 2. ¿Tiene ABATIDORES? Calcular su costo (SOLO PARA TÉRMICAS) ---
        
        # Obtenemos la tecnología de la combinación
        tec = m.tecnologia[i]
        
        # ESTE ES EL FILTRO QUE FALTABA
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl']:
            costo_mp = 0
            costo_sox = 0
            costo_nox = 0

            abatidor_mp = m.abatidor_mp[i]
            if isinstance(abatidor_mp, str):
                costo_mp = dic_equipo['MP'][abatidor_mp]['Inversión_($/kW)']

            abatidor_sox = m.abatidor_sox[i]
            if isinstance(abatidor_sox, str):
                costo_sox = dic_equipo['SOx'][abatidor_sox]['Inversión_($/kW)']

            abatidor_nox = m.abatidor_nox[i]
            if isinstance(abatidor_nox, str):
                costo_nox = dic_equipo['NOx'][abatidor_nox]['Inversión_($/kW)'] 
            
            abatidores_cost_bruto = costo_mp + costo_sox + costo_nox
            
            if abatidores_cost_bruto > 0:
                costo_anual_abatidores_por_kw = abatidores_cost_bruto * anualizacion_abatidores

        # --- 3. Construir la expresión final ---
        # Suma el costo de la central (si es nueva) + el costo de abatidores (si es térmica)
        total_fijo_expression += potencia_instalada_kw * (costo_anual_central_por_kw + costo_anual_abatidores_por_kw)
    
    return total_fijo_expression

# Costo Social por Contaminación
def costo_social(m):
    """
    Calcula el costo total por emisiones de NOx, SOx y MP,
    internalizando el costo social de la contaminación.
    """
    costo_total_social = 0
    for i in m.I:
        tec = m.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                # Calcular la energía de combustible consumida en GWh
                # m.E[i,b] está en GWh, m.eficiencia[i] es p.u.
                energia_combustible_gwh = m.E[i, b] / m.eficiencia[i]
                
                # --- Contaminante: NOx ---
                abatidor_nox = m.abatidor_nox[i]
                efi_aba_nox = 0
                if isinstance(abatidor_nox, str):
                    efi_aba_nox = dic_equipo['NOx'][abatidor_nox]['Eficiencia_(p.u.)']
                
                # Toneladas emitidas = (GWh combustible) * (ton/GWh) * (1 - abatimiento)
                toneladas_nox = energia_combustible_gwh * m.factor_emision_Nox[i] * (1 - efi_aba_nox)
                costo_total_social += toneladas_nox * m.costo_social_nox[i]
                
                # --- Contaminante: SOx ---
                abatidor_sox = m.abatidor_sox[i]
                efi_aba_sox = 0
                if isinstance(abatidor_sox, str):
                    efi_aba_sox = dic_equipo['SOx'][abatidor_sox]['Eficiencia_(p.u.)']
                
                toneladas_sox = energia_combustible_gwh * m.factor_emision_Sox[i] * (1 - efi_aba_sox)
                costo_total_social += toneladas_sox * m.costo_social_sox[i]

                # --- Contaminante: MP ---
                abatidor_mp = m.abatidor_mp[i]
                efi_aba_mp = 0
                if isinstance(abatidor_mp, str):
                    efi_aba_mp = dic_equipo['MP'][abatidor_mp]['Eficiencia_(p.u.)']
                
                toneladas_mp = energia_combustible_gwh * m.modelo_emision_MP[i] * (1 - efi_aba_mp)
                costo_total_social += toneladas_mp * m.costo_social_mp[i]

    return costo_total_social

# %%

r_df = 0.01
df_2016_2030 = 1 / (1 + r_df)**(year)

# SOLUCIÓN
def objective_rule(m):
    costo_total_operacion = costo_operacion(m) # Llama a tu función
    costo_total_fijo = costo_fijo(m)           # Llama a tu función
    #costo_total_social = costo_social(m)       # Llama a la nueva función de costo social
    return df_2016_2030 * (costo_total_operacion+ costo_total_fijo)

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

#%% 

# Hacemos un loop para resolver el modelo cambiando los porcentajes de reducción de CO2

metas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
resultados_resumen = []
emisiones_bau_co2 = 21290785.35  # ton CO2 (valor de referencia BAU)

# Parametros del solver
solver = pyo.SolverFactory('highs')
logpath = r"c:\Users\DiegoYera\Desktop\2025-2\Econo_Med\Tareas_Economed\T2\highs_log_CON_norma.log"
solver.options["log_file"] = logpath
solver.options['mip_rel_gap'] = tolerancia

# Para los costos marginales, necesitamos habilitar los multiplicadores duales
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

#%%

# --- DEFINICIÓN DE LA EXPRESIÓN DE EMISIONES (Reutilizable) ---
def get_emisiones_expression(m):
    # Esta función devuelve la EXPRESIÓN matemática (suma de variables), no el valor numérico
    suma_emisiones = 0
    for i in m.I:
        tec = m.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                # E[i,b] es la variable
                suma_emisiones += (m.E[i, b] / m.eficiencia[i]) * m.factor_emision_CO2[i]
    return suma_emisiones

# %% [BLOQUE DE EJECUCIÓN Y GENERACIÓN DE TABLAS - PREGUNTA 1]


# --- 0. PREPARACIÓN ---
print("\n--- INICIANDO EJECUCIÓN PARA TABLAS 1.1 a 1.5 ---")

resultados_completos = []
emisiones_bau_co2 = 0 
factor_conversion_2030 = 1 / df_2016_2030 # Para llevar costos de la FO (2016) a 2030

# Función auxiliar para la restricción
def get_emisiones_expression(m):
    suma = 0
    for i in m.I:
        tec = m.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                suma += (m.E[i, b] / m.eficiencia[i]) * m.factor_emision_CO2[i]
    return suma

# --- 1. LOOP DE RESOLUCIÓN ---
for meta in metas:
    print(f"\n>>> Procesando Meta: {meta*100:.0f}%")
    
    # A. Gestión de la Restricción
    if hasattr(model, 'meta_co2_constraint'):
        model.del_component(model.meta_co2_constraint)
    
    if meta == 0.0:
        print(f"    Modo BAU: Sin restricción activa")
    else:
        limite_actual = emisiones_bau_co2 * (1 - meta)
        model.meta_co2_constraint = pyo.Constraint(expr= get_emisiones_expression(model) <= limite_actual)
        print(f"    Restricción activa: <= {limite_actual:,.2f} ton")

    # B. Resolver
    results = solver.solve(model)
    
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"    ⚠️ NO ÓPTIMO para meta {meta}")
        continue 

    # C. Extracción de Datos
    # C.1 Costos del Sistema (Inv + Op + Falla)
    # Nota: model.obj está descontado al 2016. Lo llevamos a 2030.
    costo_sistema_2030 = pyo.value(model.obj) * factor_conversion_2030
    
    # C.2 Costo Marginal (Dual)
    cmg_co2 = 0
    if meta > 0 and hasattr(model, 'meta_co2_constraint') and model.meta_co2_constraint in model.dual:
        # El dual viene en valor presente (2016), lo llevamos a 2030
        cmg_co2 = model.dual[model.meta_co2_constraint] * factor_conversion_2030

    # C.3 Desglose de Emisiones y Daños
    datos_iter = {
        'Emisiones_CO2': 0, 'Emisiones_MP': 0, 'Emisiones_SOx': 0, 'Emisiones_NOx': 0,
        'Dano_MP': 0, 'Dano_SOx': 0, 'Dano_NOx': 0
    }
    
    for i in model.I:
        tec = model.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and model.eficiencia[i] > 0:
            for b in model.B:
                e_gen = pyo.value(model.E[i, b])
                if e_gen > 1e-6:
                    e_comb = e_gen / model.eficiencia[i]
                    
                    # CO2
                    co2_ton = e_comb * model.factor_emision_CO2[i]
                    datos_iter['Emisiones_CO2'] += co2_ton
                    
                    # Locales (MP, SOx, NOx)
                    for cont, param_ems, param_costo, param_abat in [
                        ('MP', model.modelo_emision_MP, model.costo_social_mp, model.abatidor_mp),
                        ('SOx', model.factor_emision_Sox, model.costo_social_sox, model.abatidor_sox),
                        ('NOx', model.factor_emision_Nox, model.costo_social_nox, model.abatidor_nox)
                    ]:
                        # Eficiencia abatidor
                        abat_nom = param_abat[i]
                        efi = 0
                        if isinstance(abat_nom, str):
                            key = 'MP' if cont == 'MP' else cont
                            if abat_nom in dic_equipo[key]:
                                efi = dic_equipo[key][abat_nom]['Eficiencia_(p.u.)']
                        
                        emis = e_comb * param_ems[i] * (1 - efi)
                        costo = emis * param_costo[i] # $/ton * ton = $
                        
                        datos_iter[f'Emisiones_{cont}'] += emis
                        datos_iter[f'Dano_{cont}'] += costo

    # C.4 Daño CO2 ($50/ton)
    dano_co2 = datos_iter['Emisiones_CO2'] * 50
    
    # Guardar referencia BAU
    if meta == 0.0:
        emisiones_bau_co2 = datos_iter['Emisiones_CO2']

    # Guardar todo en la lista
    resultados_completos.append({
        'Meta': meta,
        'Costo_Sistema_2030': costo_sistema_2030,
        'Costo_Marginal': abs(cmg_co2),
        'Emis_MP': datos_iter['Emisiones_MP'],
        'Emis_SOx': datos_iter['Emisiones_SOx'],
        'Emis_NOx': datos_iter['Emisiones_NOx'],
        'Emis_CO2': datos_iter['Emisiones_CO2'],
        'Dano_MP': datos_iter['Dano_MP'],
        'Dano_SOx': datos_iter['Dano_SOx'],
        'Dano_NOx': datos_iter['Dano_NOx'],
        'Dano_CO2': dano_co2,
        'Dano_Total': datos_iter['Dano_MP'] + datos_iter['Dano_SOx'] + datos_iter['Dano_NOx'] + dano_co2
    })

print("--- FIN DEL LOOP ---")

# --- 2. GENERACIÓN DE TABLAS (FORMATO PDF) ---

df_raw = pd.DataFrame(resultados_completos)
df_raw['Política (%)'] = (df_raw['Meta'] * 100).astype(int).astype(str) + '%'

# TABLA 1.1: Emisiones no mitigadas [Miles de Ton] [cite: 304]
t1_1 = pd.DataFrame()
t1_1['Política (%)'] = df_raw['Política (%)']
t1_1['Emisiones MP (Miles Ton)'] = df_raw['Emis_MP'] / 1000
t1_1['Emisiones SOx (Miles Ton)'] = df_raw['Emis_SOx'] / 1000
t1_1['Emisiones NOx (Miles Ton)'] = df_raw['Emis_NOx'] / 1000
t1_1['Emisiones CO2 (Miles Ton)'] = df_raw['Emis_CO2'] / 1000

# TABLA 1.2: Costo anual de mitigación [cite: 309]
costo_bau = df_raw.loc[0, 'Costo_Sistema_2030']
t1_2 = pd.DataFrame()
t1_2['Política (%)'] = df_raw['Política (%)']
t1_2['Costo (1) [US$]'] = df_raw['Costo_Sistema_2030']
t1_2['Costos solo por Mitigación (2) [US$]'] = df_raw['Costo_Sistema_2030'] - costo_bau

# TABLA 1.3: Daño ambiental y climático [cite: 314]
t1_3 = pd.DataFrame()
t1_3['Política (%)'] = df_raw['Política (%)']
t1_3['Daño MP [US$]'] = df_raw['Dano_MP']
t1_3['Daño SOx [US$]'] = df_raw['Dano_SOx']
t1_3['Daño NOx [US$]'] = df_raw['Dano_NOx']
t1_3['Daño CO2 [US$]'] = df_raw['Dano_CO2']
t1_3['Daño (3) [US$]'] = df_raw['Dano_Total']

# TABLA 1.4: Costos totales [cite: 319]
t1_4 = pd.DataFrame()
t1_4['Política (%)'] = df_raw['Política (%)']
t1_4['Costo (4) [US$]'] = t1_2['Costo (1) [US$]'] # Es el mismo
t1_4['Daño (5) [US$]'] = t1_3['Daño (3) [US$]']   # Es el mismo
t1_4['Total [US$]'] = t1_4['Costo (4) [US$]'] + t1_4['Daño (5) [US$]']

# TABLA 1.5: Costos medios y marginales [cite: 324]
t1_5 = pd.DataFrame()
t1_5['Política (%)'] = df_raw['Política (%)']

# Calculos incrementales (Fila actual - Fila anterior)
# Mitigación incremental = Emisiones CO2 Anterior - Emisiones CO2 Actual
emis_co2 = df_raw['Emis_CO2']
mitigacion_inc = -emis_co2.diff() # Negativo porque al bajar meta, bajamos emisiones
mitigacion_inc[0] = 0 # BAU no mitiga respecto a nada anterior en la tabla

# Costo Incremental = Costo Actual - Costo Anterior
costo_sys = df_raw['Costo_Sistema_2030']
costo_inc = costo_sys.diff()
costo_inc[0] = 0

t1_5['Mitigación Incremental (6) [Miles Ton]'] = mitigacion_inc / 1000
t1_5['Costo Incremental (7) [US$]'] = costo_inc

# Costo Medio Incremental ($ / Ton) = ($ Incremental) / (Ton Mitigadas)
# Ojo: Ton Mitigadas = Miles Ton * 1000
mitigacion_ton = mitigacion_inc # ya está en ton
# Evitamos división por cero en la primera fila
t1_5['Costo Medio Incremental [US$/Ton]'] = costo_inc / mitigacion_ton
t1_5.loc[0, 'Costo Medio Incremental [US$/Ton]'] = 0

t1_5['Costo Marginal [US$/Ton]'] = df_raw['Costo_Marginal']

# --- 3. EXPORTACIÓN A EXCEL (Multi-hoja) ---
archivo_salida = "Tablas_Tarea3_Pregunta1.xlsx"
with pd.ExcelWriter(archivo_salida) as writer:
    t1_1.to_excel(writer, sheet_name='Tabla 1.1', index=False)
    t1_2.to_excel(writer, sheet_name='Tabla 1.2', index=False)
    t1_3.to_excel(writer, sheet_name='Tabla 1.3', index=False)
    t1_4.to_excel(writer, sheet_name='Tabla 1.4', index=False)
    t1_5.to_excel(writer, sheet_name='Tabla 1.5', index=False)

print("\n" + "="*60)
print(f"✅ ¡LISTO! Se ha generado el archivo '{archivo_salida}'")
print("   Contiene las 5 tablas exactas pedidas en el PDF.")
print("="*60)

# Imprimir vista previa en consola
print("\n--- Vista Previa Tabla 1.1 (Emisiones) ---")
print(t1_1)
print("\n--- Vista Previa Tabla 1.5 (Costos Medios/Marginales) ---")
print(t1_5)