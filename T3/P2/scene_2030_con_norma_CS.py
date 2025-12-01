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

# --- Carga de Datos (Igual al anterior) ---
df_equipo_mp = pd.read_csv('abatidores/equipo_mp.csv').fillna(0)
equipo_mp = df_equipo_mp.set_index(['Equipo_MP'])

df_equipo_nox = pd.read_csv('abatidores/equipo_nox.csv').fillna(0)
equipo_nox = df_equipo_nox.set_index(['Equipo_NOx'])

df_equipo_sox = pd.read_csv('abatidores/equipo_sox.csv').fillna(0)
equipo_sox = df_equipo_sox.set_index(['Equipo_SOx'])

dic_equipo = {'MP':equipo_mp.to_dict(orient='index'), 
              'NOx':equipo_nox.to_dict(orient='index'), 
              'SOx':equipo_sox.to_dict(orient='index')}

dispnibilidad_hidro = [0.8215,0.6297,0.561]
costo_falla = 500  # $/MWh
year = 2030 - 2016

def excel_range_to_df(path, sheet='existentes', cell_range='E7:AB2695', header=None, engine='openpyxl'):
    m = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', cell_range.upper())
    if not m: raise ValueError("Rango debe tener formato 'E7:AB2695'")
    c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    usecols = f"{c1}:{c2}"
    skiprows = r1 - 1
    nrows = r2 - r1 + 1
    return pd.read_excel(path, sheet_name=sheet, usecols=usecols, skiprows=skiprows, nrows=nrows, header=header, engine=engine)

def obtener_tecnologia(central):
    if central == 'central_falla': return 'central_falla'
    index = central.find('|')
    return central[:index].strip()

def fd_hidro(bloque):
    if bloque == 'bloque_1': return dispnibilidad_hidro[0]
    elif bloque == 'bloque_2': return dispnibilidad_hidro[1]
    else: return dispnibilidad_hidro[2]

def anualidad(r, n): 
    return r / (1 - (1 + r)**(-n))

excel_path = 'datos_t2.xlsx'
df_existentes = excel_range_to_df(excel_path, sheet='existentes', cell_range='E7:AB2695', header=0)

CONJ = df_existentes.set_index('id_combinacion').to_dict(orient='index')
CONJ_C = df_existentes['id_centralcomb'].unique().tolist()

# --- Preparación para Pyomo ---
MAPEO_C_a_I = defaultdict(list)
for i, data in CONJ.items():
    c = data['id_centralcomb']
    MAPEO_C_a_I[c].append(i)
MAPEO_C_a_I = dict(MAPEO_C_a_I)

dic_bloques = {'bloque_1': {'duracion': 1200 , 'demanda' : 10233.87729},
               'bloque_2': {'duracion': 4152 , 'demanda' : 7872.0103}, 
               'bloque_3': {'duracion': 3408 , 'demanda' : 6297.872136}}

central_falla = 225
tasa_descuento = 0.1

# --- Modelo Pyomo ---
model = pyo.ConcreteModel(name="Modelo_Eficiente_P2")

model.I = pyo.Set(initialize=CONJ.keys())
model.C = pyo.Set(initialize=CONJ_C)
model.B = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])

# --- Parámetros ---
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

model.costo_social_mp = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_MP($/ton)'])
model.costo_social_sox = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_Sox($/ton)'])
model.costo_social_nox = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_Nox($/ton)'])
model.costo_social_co2 = pyo.Param(model.I, initialize=lambda m, i: CONJ[i]['CS_Co2($/ton)'])

# --- Conversión de Unidades ---
poder_calorifico = {'carbon': 1/132.3056959, 'petroleo_diesel': 1/79.6808152, 'cc-gnl': 1/61.96031117}

def convertir_unidades(tec, valor_kg_ton):
    cal_val = poder_calorifico.get(tec)
    if cal_val is None or math.isnan(valor_kg_ton): return 0
    return (valor_kg_ton / 1000) / cal_val

emision_descontrolada = defaultdict(dict)
for c in model.C:
    primer_i = MAPEO_C_a_I[c][0] 
    tec = model.tecnologia[primer_i]
    for cont, col in [('NOx','Nox'),('SOx','Sox'),('MP','MP'),('CO2','CO2')]:
        emision_descontrolada[c][cont] = convertir_unidades(tec, CONJ[primer_i][f'ED_{col}(kg/Mg)'])

def cargar_factor_emision(m, i, cont_largo):
    tec = m.tecnologia[i]
    return convertir_unidades(tec, CONJ[i][f'ED_{cont_largo}(kg/Mg)'])

model.factor_emision_Nox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'Nox'))
model.factor_emision_Sox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'Sox'))
model.modelo_emision_MP = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'MP'))
model.factor_emision_CO2 = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'CO2'))

# --- Variables ---
model.P = pyo.Var(model.I, within=pyo.NonNegativeReals)
model.E = pyo.Var(model.I, model.B, within=pyo.NonNegativeReals)

# --- Restricciones Técnicas (Se mantienen igual) ---
def balance_demanda(m,b):
    generacion_total = sum(m.E[i,b] for i in m.I)
    return generacion_total* (1000/(1+perdida)) >= m.param_bloques[b]['demanda']* m.param_bloques[b]['duracion']

def potencia_existente(m, c):
    combinaciones = MAPEO_C_a_I[c]
    pot_neta = m.potencia_neta[combinaciones[0]]
    if math.isnan(pot_neta): return pyo.Constraint.Skip
    return sum(m.P[i] for i in combinaciones) == pot_neta

def disponibilidad_tecnica(m, i, b):
    if m.tecnologia[i] == 'central_falla': return pyo.Constraint.Skip
    disp = fd_hidro(b) if m.tecnologia[i] in ['hidro', 'hidro_conv', 'minihidro'] else m.disponibilidad[i]
    return (m.E[i, b] * 1000) <= m.P[i] * disp * m.param_bloques[b]['duracion']

def capacidad_por_central(m, c):
    combinaciones = MAPEO_C_a_I[c]
    pot_max = m.potencia_max[combinaciones[0]]
    if math.isnan(pot_max): return pyo.Constraint.Skip
    return sum(m.P[i] for i in combinaciones) <= pot_max

model.demanda_constraint = pyo.Constraint(model.B, rule=balance_demanda)
model.potencia_existente_constraint = pyo.Constraint(model.C, rule=potencia_existente)
model.disponibilidad_tecnica_constraint = pyo.Constraint(model.I, model.B, rule=disponibilidad_tecnica)
model.capacidad_por_central_constraint = pyo.Constraint(model.C, rule=capacidad_por_central)

# ### [MODIFICACIÓN P2] ELIMINACIÓN DE NORMAS ###

# --- Función Objetivo ---

# Costo Operación Privado
def costo_operacion(m):
    total_variable = 0
    for b in m.B:
        for i in m.I: 
            if m.tecnologia[i] == 'central_falla':
                total_variable += m.E[i, b] * 1000 * costo_falla
                continue
            
            costo_var = m.costo_variable[i]
            costo_abat = 0
            
            # Sumar costo variable de abatidores si existen
            if m.tecnologia[i] in ['carbon', 'petroleo_diesel', 'cc-gnl']:
                for tipo, param_abat in [('MP', m.abatidor_mp), ('SOx', m.abatidor_sox), ('NOx', m.abatidor_nox)]:
                    abat_nombre = param_abat[i]
                    if isinstance(abat_nombre, str):
                        costo_abat += dic_equipo[tipo][abat_nombre]['Costo_variable_($/MWh)']
            
            total_variable += m.E[i, b] * 1000 * (costo_var + costo_abat)
    return total_variable

# Costo Fijo Privado
def costo_fijo(m):
    total_fijo = 0.0
    vida_abat, r_abat = 30, 0.1
    anual_abat = anualidad(r_abat, vida_abat)

    for i in m.I:
        pot_kw = m.P[i] * 1000
        costo_kw_anual = 0.0
        
        # Inversión central nueva
        if math.isnan(m.potencia_neta[i]):
            if m.costo_fijo[i] > 0 and m.vida_util[i] > 0:
                costo_kw_anual += m.costo_fijo[i] * anualidad(tasa_descuento, m.vida_util[i])
        
        # Inversión abatidores
        if m.tecnologia[i] in ['carbon', 'petroleo_diesel', 'cc-gnl']:
            inv_abat = 0
            for tipo, param_abat in [('MP', m.abatidor_mp), ('SOx', m.abatidor_sox), ('NOx', m.abatidor_nox)]:
                abat_nombre = param_abat[i]
                if isinstance(abat_nombre, str):
                    inv_abat += dic_equipo[tipo][abat_nombre]['Inversión_($/kW)']
            if inv_abat > 0:
                costo_kw_anual += inv_abat * anual_abat
        
        total_fijo += pot_kw * costo_kw_anual
    return total_fijo

# ### COSTO SOCIAL INTERNALIZADO ###
# Impuesto al CO2 ($50/ton) y los costos sociales locales.
def costo_social_total(m):
    total_social = 0
    TAX_CO2 = 50.0  # [cite: 190] Carbon Tax igual al costo social
    
    for i in m.I:
        tec = m.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                # Energía combustible en GWh
                e_comb = m.E[i, b] / m.eficiencia[i]
                
                # 1. Costo por CO2
                emis_co2 = e_comb * m.factor_emision_CO2[i]
                total_social += emis_co2 * TAX_CO2
                
                # 2. Costo por Contaminantes Locales (MP, SOx, NOx)
                # Debemos considerar la reducción si hay abatidor instalado
                for cont, param_emision, param_costo_soc, param_abatidor in [
                    ('MP', m.modelo_emision_MP, m.costo_social_mp, m.abatidor_mp),
                    ('SOx', m.factor_emision_Sox, m.costo_social_sox, m.abatidor_sox),
                    ('NOx', m.factor_emision_Nox, m.costo_social_nox, m.abatidor_nox)
                ]:
                    abat_nombre = param_abatidor[i]
                    efi_aba = 0.0
                    if isinstance(abat_nombre, str):
                        tipo_dic = 'MP' if cont == 'MP' else cont
                        if abat_nombre in dic_equipo[tipo_dic]:
                            efi_aba = dic_equipo[tipo_dic][abat_nombre]['Eficiencia_(p.u.)']
                    
                    emis_ton = e_comb * param_emision[i] * (1 - efi_aba)
                    total_social += emis_ton * param_costo_soc[i]
                    
    return total_social

# Factor de descuento
r_df = 0.01
df_2016_2030 = 1 / (1 + r_df)**year


def objective_rule_social(m):
    return df_2016_2030 * (costo_operacion(m) + costo_fijo(m) + costo_social_total(m))

model.obj = pyo.Objective(rule=objective_rule_social, sense=pyo.minimize)

# %%
# --- RESOLUCIÓN ---

solver = pyo.SolverFactory('highs')
# solver.options["log_file"] = ... 
solver.options['mip_rel_gap'] = tolerancia

print(">>> Resolviendo Pregunta 2: Política Eficiente (Impuestos Pigouvianos)...")
results = solver.solve(model, tee=True)

# %%
# --- REPORTE DE RESULTADOS (TABLAS 2.1 y 2.2) ---

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("\n✅ Solución Óptima encontrada.")
    
    # Extraemos los valores óptimos
    costo_privado_total = pyo.value(df_2016_2030 * (costo_operacion(model) + costo_fijo(model)))
    costo_social_total_val = pyo.value(df_2016_2030 * costo_social_total(model))
    costo_generalizado = pyo.value(model.obj)
    
    # Calculamos Emisiones Totales de CO2
    emisiones_co2 = 0
    for i in model.I:
        if model.tecnologia[i] in ['carbon', 'petroleo_diesel', 'cc-gnl'] and model.eficiencia[i] > 0:
            for b in model.B:
                e_comb = pyo.value(model.E[i, b]) / model.eficiencia[i]
                emisiones_co2 += e_comb * model.factor_emision_CO2[i]

    print("\n" + "="*50)
    print("RESULTADOS PREGUNTA 2 - AÑO 2030")
    print("="*50)
    
    # Datos para Tabla 2.1 (Emisiones Mitigadas)
    # Debes comparar esto con el BAU (Pregunta 1, meta 0%)
    print(f"1. Emisiones CO2 Totales: {emisiones_co2:,.2f} ton")
    
    # Datos para Tabla 2.2 (Costos Totales)
    print(f"\n2. Desglose de Costos (Valor Presente 2016 -> convertir a 2030 dividiendo por factor si es necesario):")
    print(f"   - Costo Privado (Inv + Op): {costo_privado_total/1e6:,.2f} MMUSD")
    print(f"   - Costo Social (Daño):      {costo_social_total_val/1e6:,.2f} MMUSD")
    print(f"   - Costo TOTAL (Objetivo):   {costo_generalizado/1e6:,.2f} MMUSD")
    
    print("\nNOTA: Para las tablas, recuerda que los valores monetarios suelen pedirse en USD del año 2030.")
    print(f"      Multiplica estos valores por (1+r)^{year} o usa los valores sin descontar si el script ya lo hace.")
    print("="*50)

    # Exportar a un mini Excel
    data_p2 = {
        'Concepto': ['Emisiones CO2 (ton)', 'Costo Privado (MMUSD)', 'Costo Social (MMUSD)', 'Costo Total (MMUSD)'],
        'Valor (2030)': [
            emisiones_co2,
            (costo_privado_total / df_2016_2030) / 1e6, # Llevado a 2030
            (costo_social_total_val / df_2016_2030) / 1e6,
            (costo_generalizado / df_2016_2030) / 1e6
        ]
    }
    pd.DataFrame(data_p2).to_excel("resultados_tarea3_p2.xlsx", index=False)
    print("📄 Resultados exportados a 'resultados_tarea3_p2.xlsx'")

else:
    print("❌ No se encontró solución óptima.")