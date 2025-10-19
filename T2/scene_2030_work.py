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

# Imprimir el DataFrame para verificar
#print(df_existentes)

# pasamos a diccionario

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
# (Asumiendo que df_existentes ya está cargado y es correcto)
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

# --- INICIO DE LA LÓGICA DE CONVERSIÓN DE UNIDADES Y CÁLCULO DE NORMA ---

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
    # La primera combinación de cada central (i.e., sin abatidores) es la base
    primer_i = MAPEO_C_a_I[c][0] 
    tec = model.tecnologia[primer_i]
    emision_descontrolada[c]['NOx'] = convertir_unidades(tec, CONJ[primer_i]['ED_Nox(kg/Mg)'])
    emision_descontrolada[c]['SOx'] = convertir_unidades(tec, CONJ[primer_i]['ED_Sox(kg/Mg)'])
    emision_descontrolada[c]['MP']  = convertir_unidades(tec, CONJ[primer_i]['ED_MP(kg/Mg)'])


# 4. Funciones para cargar los parámetros (versión corregida)
def cargar_factor_emision(m, i, cont_corto, cont_largo):
    tec = m.tecnologia[i]
    valor_kg_ton = CONJ[i][f'ED_{cont_largo}(kg/Mg)']
    return convertir_unidades(tec, valor_kg_ton)

def cargar_norma_absoluta(m, i, cont_corto, cont_largo):
    id_central = CONJ[i]['id_centralcomb']
    ed_base = emision_descontrolada[id_central][cont_corto]
    porcentaje_reduccion = CONJ[i][f'Norma_{cont_largo}']
    
    if math.isnan(porcentaje_reduccion):
        return float('inf') # Si no hay norma, el límite es infinito
        
    return ed_base * (1 - porcentaje_reduccion)

# 5. Carga final de parámetros (con las llamadas corregidas)
model.factor_emision_Nox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'NOx', 'Nox'))
model.factor_emision_Sox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'SOx', 'Sox'))
model.modelo_emision_MP = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'MP', 'MP'))

model.norma_emision_Nox = pyo.Param(model.I, initialize=lambda m, i: cargar_norma_absoluta(m, i, 'NOx', 'Nox'))
model.norma_emision_Sox = pyo.Param(model.I, initialize=lambda m, i: cargar_norma_absoluta(m, i, 'SOx', 'Sox'))
model.norma_emision_MP = pyo.Param(model.I, initialize=lambda m, i: cargar_norma_absoluta(m, i, 'MP', 'MP'))

# --- VARIABLES ---
model.P = pyo.Var(model.I, within=pyo.NonNegativeReals)            # potencia instalada [MW] por combinacion
model.E = pyo.Var(model.I, model.B, within=pyo.NonNegativeReals)  # energia generada [MWh] en el bloque


# %%

# Restricción 1: Balance de demanda por bloque
def balance_demanda(m,b):
    generacion_total = sum(m.E[i,b] for i in m.I)
    return generacion_total* (1000/(1+perdida)) >= m.param_bloques[b]['demanda']* m.param_bloques[b]['duracion']

# Restriccion 2: Se mantienen la potencia las centrales existentes
def potencia_existente(m, c):
    combinaciones_de_c = MAPEO_C_a_I[c] # equivalente a i = 64 * (c - 1) + 1 
    primer_i = combinaciones_de_c[0]
    pot_neta = m.potencia_neta[primer_i]
    if math.isnan(pot_neta):
        return pyo.Constraint.Skip
    else:
        # CORRECCIÓN: Usar la variable 'pot_neta' directamente
        return sum(m.P[i] for i in combinaciones_de_c) == pot_neta

# Restriccion 3: Dispobibilidad Técnica (maxima generacion)
def disponibilidad_tecnica(m, c, b):
    combinaciones_de_c = MAPEO_C_a_I[c] # equivalente a i = 64 * (c - 1) + 1 
    primer_i = combinaciones_de_c[0] 

    if m.tecnologia[primer_i] == 'central_falla':
        return pyo.Constraint.Skip

    if m.tecnologia[primer_i] in ['hidro', 'hidro_conv', 'minihidro']: # Usar 'in' para listas
        disp = fd_hidro(b)
    else:
        disp = m.disponibilidad[primer_i]

    generacion_central = sum(m.E[i, b] for i in combinaciones_de_c) #GWh
    potencia_instalada_central = sum(m.P[i] for i in combinaciones_de_c) #MW

    return (generacion_central * 1000) <= potencia_instalada_central * disp * m.param_bloques[b]['duracion']

# Restriccion 4: Capacidad Por Central (esto seria para las nuevas)
def capacidad_por_central(m, c):
    combinaciones_de_c = MAPEO_C_a_I[c] # equivalente a i = 64 * (c - 1) + 1 
    primer_i = combinaciones_de_c[0]
    pot_max = m.potencia_max[primer_i] # Obtener el valor primero
    if math.isnan(pot_max):
        return pyo.Constraint.Skip
    else:
        # CORRECCIÓN: Usar la variable 'pot_max'
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
            # El resto de la restricción está perfecto, las unidades ya son consistentes
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
            # El resto de la restricción está perfecto, las unidades ya son consistentes
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
            # El resto de la restricción está perfecto, las unidades ya son consistentes
            energia_combustible = m.E[i,b] / efi_calor
            return energia_combustible * norma >= energia_combustible * ed * (1 - efi_aba)
    else:
        return pyo.Constraint.Skip


model.demanda_constraint = pyo.Constraint(model.B, rule=balance_demanda)
model.potencia_existente_constraint = pyo.Constraint(model.C, rule=potencia_existente)
model.disponibilidad_tecnica_constraint = pyo.Constraint(model.C, model.B, rule=disponibilidad_tecnica)
model.capacidad_por_central_constraint = pyo.Constraint(model.C, rule=capacidad_por_central)
model.norma_emision_nox_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_nox)
model.norma_emision_sox_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_sox)
model.norma_emision_mp_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_mp)

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
            
            abatidor_mp = m.abatidor_mp[i]
            if isinstance(abatidor_mp, str):
                costo_mp = dic_equipo['MP'][abatidor_mp]['Costo_variable_($/MWh)']

            abatidor_sox = m.abatidor_sox[i]
            if isinstance(abatidor_sox, str):
                costo_sox = dic_equipo['SOx'][abatidor_sox]['Costo_variable_($/MWh)']

            abatidor_nox = m.abatidor_nox[i]
            if isinstance(abatidor_nox, str):
                costo_nox = dic_equipo['NOx'][abatidor_nox]['Costo_variable_($/MWh)']
            
            abatidores_cost = costo_mp + costo_sox + costo_nox # $/MWh

            # Energía está en GWh y los costos en $/MWh
            total_variable += m.E[i, b] * 1000 * (costo_var + abatidores_cost)
            
    return total_variable
        
# Costo Fijo        
def costo_fijo(m):
    total_fijo = 0 
    for i in m.I:
        # Costo de inversión para centrales NUEVAS
        if math.isnan(m.potencia_neta[i]):
            
            costo_inv_central = m.costo_fijo[i] # ($/kW)
            
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

            abatidores_cost = costo_mp + costo_sox + costo_nox
            
            costo_fijo_total_combinacion = costo_inv_central + abatidores_cost # $/kW
            
            # La potencia instalada P[i] está en MW, se pasa a kW
            potencia_instalada_kw = m.P[i] * 1000 
            
            vida_util = m.vida_util[i]
            if vida_util > 0:
                anualizacion = anualidad(tasa_descuento, vida_util)
                total_fijo += potencia_instalada_kw * costo_fijo_total_combinacion * anualizacion
    
    return total_fijo

# --- AÑADE ESTA NUEVA FUNCIÓN A TU CÓDIGO ---

# Costo Social por Contaminación
def costo_social(m):
    """
    Calcula el costo total por emisiones de NOx, SOx y MP,
    internalizando el costo social de la contaminación.
    """
    costo_total_social = 0
    
    # Iteramos sobre todas las combinaciones y bloques
    for i in m.I:
        # Solo aplica a tecnologías térmicas que tienen un factor de eficiencia
        tec = m.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                # 1. Calcular la energía de combustible consumida en GWh
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
    costo_total_social = costo_social(m)       # Llama a la nueva función de costo social
    
    return df_2016_2030 * (costo_total_operacion + costo_total_fijo + costo_total_social)

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

#%% 
# Resolver el modelo
solver = pyo.SolverFactory('highs')
solver.options['mip_rel_gap'] = tolerancia
results = solver.solve(model, tee=True)
print(f"Status: {results}")


# %%

# %%
# --- SCRIPT PARA MOSTRAR LA SOLUCIÓN DE GENERACIÓN (E) ---

print("\n" + "="*50)
print("⚡ REPORTE DE GENERACIÓN DE ENERGÍA (GWh) ⚡")
print("="*50 + "\n")

# Verifica si el solver encontró una solución óptima
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    
    # Un diccionario para agrupar la generación por central y por bloque
    generacion_agrupada = defaultdict(float)
    
    # Itera sobre todas las variables de energía E[i, b]
    for i, b in model.E:
        energia_generada = pyo.value(model.E[i, b])
        
        # Solo procesamos si la generación es mayor a un valor pequeño
        if energia_generada > 1e-6:
            # Obtenemos el nombre de la central desde el diccionario de datos CONJ
            nombre_central = CONJ[i]['Central']
            
            # Agregamos la energía al total de esa central en ese bloque
            generacion_agrupada[(nombre_central, b)] += energia_generada
            
    if not generacion_agrupada:
        print("No se encontró generación de energía significativa en la solución.")
    else:
        # Imprime los resultados agrupados de forma ordenada
        print(f"{'CENTRAL':<40} | {'BLOQUE':<12} | {'GENERACIÓN (GWh)':>20}")
        print("-" * 80)
        # Ordenamos los resultados por nombre de central y luego por bloque
        for (nombre, bloque), total_energia in sorted(generacion_agrupada.items()):
            print(f"{nombre:<40} | {bloque:<12} | {total_energia:>20.4f}")

else:
    print("El modelo no encontró una solución óptima.")
    print(f"Estado del Solver: {results.solver.termination_condition}")

print("\n" + "="*50)
print("FIN DEL REPORTE")
print("="*50)

# %% 

# %%
# Tras resolver: intenta cargar explícitamente la solución (por si acaso)
try:
    model.solutions.load_from(results)
except:
    pass

print("\n" + "="*50)
print("REPORTE DE POTENCIA INSTALADA (MW)")
print("="*50 + "\n")

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    potencia_instalada = []
    no_inicializadas = 0

    for i in model.P:
        val = model.P[i].value   # None si no fue parte del LP/solución
        if val is None:
            no_inicializadas += 1
            continue
        if val > 1e-6:
            info = {
                'id': i,
                'central': CONJ[i]['Central'],
                'potencia': val,
                'abatidor_mp': CONJ[i]['MP'],
                'abatidor_sox': CONJ[i]['Sox'],
                'abatidor_nox': CONJ[i]['Nox']
            }
            potencia_instalada.append(info)

    if not potencia_instalada:
        print("No se encontró instalación de potencia significativa en la solución.")
    else:
        print(f"{'ID':<6} | {'CENTRAL':<40} | {'POTENCIA (MW)':>15} | {'ABATIDORES (MP, SOx, NOx)':<55}")
        print("-" * 125)
        for item in sorted(potencia_instalada, key=lambda x: x['central']):
            abatidores_str = f"{item['abatidor_mp']}, {item['abatidor_sox']}, {item['abatidor_nox']}"
            print(f"{item['id']:<6} | {item['central']:<40} | {item['potencia']:>15.4f} | {abatidores_str:<55}")

    print(f"\nVariables P sin valor (no usadas en el LP activo): {no_inicializadas}")

else:
    print("El modelo no encontró una solución óptima.")
    print(f"Estado del Solver: {results.solver.termination_condition}")

# %%
