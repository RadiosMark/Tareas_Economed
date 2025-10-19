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
#model.norma_emision_nox_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_nox)
#model.norma_emision_sox_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_sox)
#model.norma_emision_mp_constraint = pyo.Constraint(model.I, model.B, rule=norma_emision_mp)

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
    #costo_total_social = costo_social(m)       # Llama a la nueva función de costo social

    return df_2016_2030 * (costo_total_operacion + costo_total_fijo)

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

#%% 
# Resolver el modelo
solver = pyo.SolverFactory('highs')
solver.options['mip_rel_gap'] = tolerancia
results = solver.solve(model, tee=True)
print(f"Status: {results}")


# %%
# --- SCRIPT PARA GENERAR REPORTE COMPLETO EN UN ARCHIVO .TXT ---

print("📝 Generando reporte de resultados...")

# Nombre del archivo de salida
nombre_archivo = 'resultados_modelo_base.txt'

# --- INICIO DEL SCRIPT ---
try:
    # Intenta cargar la solución en el modelo para asegurar que los valores estén disponibles
    model.solutions.load_from(results)
except (ValueError, AttributeError):
    print("⚠️ Advertencia: No se pudo cargar la solución en el modelo. Los resultados pueden no estar disponibles si el modelo es infactible.")

# Verifica si el solver encontró una solución óptima
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    
    # --- CÁLCULOS PREVIOS ---
    
    # 1. Agrupar potencia por planta y tecnología
    potencia_por_planta = defaultdict(float)
    potencia_por_tecnologia = defaultdict(float)
    for i in model.P:
        potencia_valor = model.P[i].value
        if potencia_valor is not None and potencia_valor > 1e-6:
            nombre_central = CONJ[i]['Central']
            tecnologia = model.tecnologia[i]
            potencia_por_planta[nombre_central] += potencia_valor
            if tecnologia != 'central_falla':
                potencia_por_tecnologia[tecnologia] += potencia_valor

    # 2. Agrupar generación por tecnología
    generacion_por_tecnologia = defaultdict(float)
    for i, b in model.E:
        energia_generada = model.E[i, b].value
        if energia_generada is not None and energia_generada > 1e-6:
            tecnologia = model.tecnologia[i]
            if tecnologia != 'central_falla':
                generacion_por_tecnologia[tecnologia] += energia_generada

    # 3. Calcular emisiones totales
    emisiones = {'MP': 0, 'SOx': 0, 'NOx': 0, 'CO2': 0}
    # Asegúrate de tener cargado el parámetro de emisión de CO2
    model.factor_emision_CO2 = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['ED_CO2(kg/Mg)'])
    
    for i in model.I:
        tec = model.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and model.eficiencia[i] > 0:
            for b in model.B:
                energia_gen_gwh = model.E[i, b].value
                if energia_gen_gwh is not None and energia_gen_gwh > 1e-6:
                    energia_combustible_gwh = energia_gen_gwh / model.eficiencia[i]
                    
                    # Cálculo para MP, SOx, NOx (ya están en ton/GWh)
                    for cont_corto, cont_largo, param_emision in [('MP', 'MP', model.modelo_emision_MP), ('SOx', 'Sox', model.factor_emision_Sox), ('NOx', 'Nox', model.factor_emision_Nox)]:
                        abatidor = getattr(model, f'abatidor_{cont_largo.lower()}')[i]
                        efi_aba = 0
                        if isinstance(abatidor, str):
                            efi_aba = dic_equipo[cont_corto][abatidor]['Eficiencia_(p.u.)']
                        toneladas_emitidas = energia_combustible_gwh * param_emision[i] * (1 - efi_aba)
                        emisiones[cont_corto] += toneladas_emitidas
                    
                    # Cálculo para CO2 (necesita conversión)
                    co2_kg_ton = model.factor_emision_CO2[i]
                    co2_ton_gwh = convertir_unidades(tec, co2_kg_ton)
                    emisiones['CO2'] += energia_combustible_gwh * co2_ton_gwh
    
    # 4. Calcular el Costo Social Total de este escenario (para el beneficio de la norma)
    costo_social_actual_total = 0
    for i in model.I:
        tec = model.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and model.eficiencia[i] > 0:
            for b in model.B:
                # Reutilizamos los cálculos de emisiones
                energia_combustible_gwh = pyo.value(model.E[i, b] / model.eficiencia[i])
                if energia_combustible_gwh > 1e-6:
                    # NOx
                    efi_aba_nox = 0
                    if isinstance(model.abatidor_nox[i], str): efi_aba_nox = dic_equipo['NOx'][model.abatidor_nox[i]]['Eficiencia_(p.u.)']
                    costo_social_actual_total += (energia_combustible_gwh * model.factor_emision_Nox[i] * (1 - efi_aba_nox)) * model.costo_social_nox[i]
                    # SOx
                    efi_aba_sox = 0
                    if isinstance(model.abatidor_sox[i], str): efi_aba_sox = dic_equipo['SOx'][model.abatidor_sox[i]]['Eficiencia_(p.u.)']
                    costo_social_actual_total += (energia_combustible_gwh * model.factor_emision_Sox[i] * (1 - efi_aba_sox)) * model.costo_social_sox[i]
                    # MP
                    efi_aba_mp = 0
                    if isinstance(model.abatidor_mp[i], str): efi_aba_mp = dic_equipo['MP'][model.abatidor_mp[i]]['Eficiencia_(p.u.)']
                    costo_social_actual_total += (energia_combustible_gwh * model.modelo_emision_MP[i] * (1 - efi_aba_mp)) * model.costo_social_mp[i]


    # --- ESCRITURA DEL ARCHIVO .TXT ---
    
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                  REPORTE DE RESULTADOS DEL MODELO DE OPTIMIZACIÓN\n")
        f.write("="*80 + "\n\n")

        # --- 4) COSTO TOTAL DEL SISTEMA ---
        costo_total = pyo.value(model.obj)
        f.write("--- 4. Costo Total del Sistema ---\n")
        f.write(f"Costo Óptimo: {costo_total / 1e6:.4f} MMUS$\n\n")
        
        # --- 5) BENEFICIO DE LA NORMA ---
        f.write("--- 5. Beneficio de la Norma ---\n")
        f.write(f"Costo Social de la Contaminación en este escenario: {costo_social_actual_total / 1e6:.4f} MMUS$\n")
        f.write("NOTA: Para calcular el 'Beneficio de la Norma', corre el modelo 'Sin Norma' y 'Con Norma'.\n")
        f.write("      Luego, calcula: Beneficio = (Costo Social Sin Norma) - (Costo Social Con Norma).\n\n")

        # --- 6) EMISIONES TOTALES ---
        f.write("--- 6. Emisiones Totales (centrales térmicas) ---\n")
        f.write(f"{'Contaminante':<15} | {'Emisiones (toneladas)':>25}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MP':<15} | {emisiones['MP']:>25.2f}\n")
        f.write(f"{'SOx':<15} | {emisiones['SOx']:>25.2f}\n")
        f.write(f"{'NOx':<15} | {emisiones['NOx']:>25.2f}\n")
        f.write(f"{'CO2':<15} | {emisiones['CO2']:>25.2f}\n\n")

        # --- 1) POTENCIA POR PLANTA ---
        f.write("--- 1. Potencia Instalada por Planta (MW) ---\n")
        f.write(f"{'Planta':<40} | {'Potencia (MW)':>15}\n")
        f.write("-" * 60 + "\n")
        for planta, potencia in sorted(potencia_por_planta.items()):
            f.write(f"{planta:<40} | {potencia:>15.2f}\n")
        f.write("\n")

        # --- 2) POTENCIA POR TECNOLOGÍA ---
        f.write("--- 2. Potencia Instalada por Tecnología (MW) ---\n")
        f.write(f"{'Tecnología':<25} | {'Potencia (MW)':>20}\n")
        f.write("-" * 50 + "\n")
        for tec, potencia in sorted(potencia_por_tecnologia.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{tec:<25} | {potencia:>20.2f}\n")
        f.write("\n")

        # --- 3) GENERACIÓN POR TECNOLOGÍA ---
        f.write("--- 3. Generación Anual por Tecnología (GWh) ---\n")
        f.write(f"{'Tecnología':<25} | {'Generación (GWh)':>20}\n")
        f.write("-" * 50 + "\n")
        for tec, energia in sorted(generacion_por_tecnologia.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{tec:<25} | {energia:>20.2f}\n")
    
    print(f"✅ Reporte guardado exitosamente en el archivo '{nombre_archivo}'")

else:
    print("❌ El modelo no encontró una solución óptima. No se generó el reporte.")
    print(f"   Estado del Solver: {results.solver.termination_condition}")

# %%


