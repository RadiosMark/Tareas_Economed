# %% 
import pandas as pd
import re
import highspy as hgs
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from collections import defaultdict
import math

# --- 1. CONFIGURACIÓN Y DATOS ---
print("--- INICIANDO MODELO PREGUNTA 2 (IMPUESTOS PIGOUVIANOS) ---")

tolerancia = 0.00001
perdida = 0.04
tasa_descuento = 0.1
year = 2030 - 2016 # Factor para llevar costos de 2016 a 2030
r_df = 0.01 # Tasa de descuento 
factor_conversion_2030 = (1 + r_df)**year 
df_2016_2030_inv = 1 / factor_conversion_2030 # Factor descuento para la FO

t_centrales = ['biomasa', 'carbon','cc-gnl', 'petroleo_diesel', 'hidro', 'minihidro','eolica','solar', 'geotermia']

# Carga de Abatidores
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

# Funciones de carga
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

# Carga Excel
excel_path = 'datos_t2.xlsx'
df_existentes = excel_range_to_df(excel_path, sheet='existentes', cell_range='E7:AB2695', header=0)

CONJ = df_existentes.set_index('id_combinacion').to_dict(orient='index')
CONJ_C = df_existentes['id_centralcomb'].unique().tolist()

# Mapeos
MAPEO_C_a_I = defaultdict(list)
for i, data in CONJ.items():
    c = data['id_centralcomb']
    MAPEO_C_a_I[c].append(i)
MAPEO_C_a_I = dict(MAPEO_C_a_I)

dic_bloques = {'bloque_1': {'duracion': 1200 , 'demanda' : 10233.87729},
               'bloque_2': {'duracion': 4152 , 'demanda' : 7872.0103}, 
               'bloque_3': {'duracion': 3408 , 'demanda' : 6297.872136}}

central_falla = 225

# --- 2. MODELO PYOMO ---
model = pyo.ConcreteModel(name="Modelo_Eficiente_P2")

model.I = pyo.Set(initialize=CONJ.keys())
model.C = pyo.Set(initialize=CONJ_C)
model.B = pyo.Set(initialize=['bloque_1','bloque_2','bloque_3'])

# Parámetros
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

# Conversión Factores Emisión
poder_calorifico = {'carbon': 1/132.3056959, 'petroleo_diesel': 1/79.6808152, 'cc-gnl': 1/61.96031117}

def convertir_unidades(tec, valor_kg_ton):
    cal_val = poder_calorifico.get(tec)
    if cal_val is None or math.isnan(valor_kg_ton): return 0
    return (valor_kg_ton / 1000) / cal_val

def cargar_factor_emision(m, i, cont_largo):
    tec = m.tecnologia[i]
    return convertir_unidades(tec, CONJ[i][f'ED_{cont_largo}(kg/Mg)'])

model.factor_emision_Nox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'Nox'))
model.factor_emision_Sox = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'Sox'))
model.modelo_emision_MP = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'MP'))
model.factor_emision_CO2 = pyo.Param(model.I, initialize=lambda m, i: cargar_factor_emision(m, i, 'CO2'))

# Variables
model.P = pyo.Var(model.I, within=pyo.NonNegativeReals)
model.E = pyo.Var(model.I, model.B, within=pyo.NonNegativeReals)

# Restricciones
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

# --- 3. FUNCIONES DE COSTO (INCLUYENDO SOCIAL) ---

def costo_operacion(m):
    total_variable = 0
    for b in m.B:
        for i in m.I: 
            if m.tecnologia[i] == 'central_falla':
                total_variable += m.E[i, b] * 1000 * costo_falla
                continue
            costo_var = m.costo_variable[i]
            costo_abat = 0
            if m.tecnologia[i] in ['carbon', 'petroleo_diesel', 'cc-gnl']:
                for tipo, param_abat in [('MP', m.abatidor_mp), ('SOx', m.abatidor_sox), ('NOx', m.abatidor_nox)]:
                    abat_nombre = param_abat[i]
                    if isinstance(abat_nombre, str):
                        costo_abat += dic_equipo[tipo][abat_nombre]['Costo_variable_($/MWh)']
            total_variable += m.E[i, b] * 1000 * (costo_var + costo_abat)
    return total_variable

def costo_fijo(m):
    total_fijo = 0.0
    vida_abat, r_abat = 30, 0.1
    anual_abat = anualidad(r_abat, vida_abat)
    for i in m.I:
        pot_kw = m.P[i] * 1000
        costo_kw_anual = 0.0
        if math.isnan(m.potencia_neta[i]):
            if m.costo_fijo[i] > 0 and m.vida_util[i] > 0:
                costo_kw_anual += m.costo_fijo[i] * anualidad(tasa_descuento, m.vida_util[i])
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

def costo_social_total(m):
    total_social = 0
    TAX_CO2 = 50.0 
    for i in m.I:
        tec = m.tecnologia[i]
        if tec in ['carbon', 'petroleo_diesel', 'cc-gnl'] and m.eficiencia[i] > 0:
            for b in m.B:
                e_comb = m.E[i, b] / m.eficiencia[i]
                # CO2 (Impuesto)
                total_social += e_comb * m.factor_emision_CO2[i] * TAX_CO2
                # Locales
                for cont, param_ems, param_costo, param_abat in [
                    ('MP', m.modelo_emision_MP, m.costo_social_mp, m.abatidor_mp),
                    ('SOx', m.factor_emision_Sox, m.costo_social_sox, m.abatidor_sox),
                    ('NOx', m.factor_emision_Nox, m.costo_social_nox, m.abatidor_nox)
                ]:
                    abat_nom = param_abat[i]
                    efi = 0
                    if isinstance(abat_nom, str):
                        key = 'MP' if cont == 'MP' else cont
                        if abat_nom in dic_equipo[key]:
                            efi = dic_equipo[key][abat_nom]['Eficiencia_(p.u.)']
                    total_social += e_comb * param_ems[i] * (1 - efi) * param_costo[i]
    return total_social

# Función Objetivo P2: Fijo + Operacion + Social
def objective_rule_social(m):
    return df_2016_2030_inv * (costo_operacion(m) + costo_fijo(m) + costo_social_total(m))

model.obj = pyo.Objective(rule=objective_rule_social, sense=pyo.minimize)

# --- 4. RESOLUCIÓN ---
solver = pyo.SolverFactory('highs')
solver.options['mip_rel_gap'] = tolerancia
print(">>> Resolviendo...")
results = solver.solve(model, tee=True)

# %% 
# --- 5. GENERACIÓN DE TABLAS (FORMATO PDF) ---

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("\n✅ Solución Óptima encontrada.")
    
    # --- RESULTADOS ÓPTIMOS P2 ---
    # Costos en valor presente 2016
    opt_cp_vp = pyo.value(costo_operacion(model) + costo_fijo(model))
    opt_cs_vp = pyo.value(costo_social_total(model))
    
    # Llevamos a 2030 para las tablas
    opt_costo_privado_2030 = opt_cp_vp * factor_conversion_2030
    opt_dano_ambiental_2030 = opt_cs_vp * factor_conversion_2030
    
    # Emisiones CO2
    opt_emisiones_co2 = 0
    for i in model.I:
        if model.tecnologia[i] in ['carbon', 'petroleo_diesel', 'cc-gnl'] and model.eficiencia[i] > 0:
            for b in model.B:
                opt_emisiones_co2 += (pyo.value(model.E[i, b]) / model.eficiencia[i]) * model.factor_emision_CO2[i]

    # -------------------------------------------------------------------------
    # Valores de BAU tabla 1.1 a 1.3 escenario 0%
    
    BAU_EMISIONES_CO2 = 21290785.350555  
    BAU_COSTO_PRIVADO = 2156143454       
    BAU_DANO_AMBIENTAL = 1257493547     
    
    print("\n--- DATOS DE ENTRADA BAU (Verificar en script) ---")
    print(f"BAU Emisiones CO2: {BAU_EMISIONES_CO2:,.2f}")
    print(f"BAU Costo Privado: {BAU_COSTO_PRIVADO:,.2f}")
    print(f"BAU Daño Ambiental:{BAU_DANO_AMBIENTAL:,.2f}")
    # -------------------------------------------------------------------------

    # CALCULOS TABLAS
    # Tabla 2.1
    mitigacion_co2_ton = BAU_EMISIONES_CO2 - opt_emisiones_co2
    mitigacion_co2_pct = (mitigacion_co2_ton / BAU_EMISIONES_CO2) * 100 if BAU_EMISIONES_CO2 > 0 else 0
    
    # Tabla 2.2
    bau_total = BAU_COSTO_PRIVADO + BAU_DANO_AMBIENTAL
    opt_total = opt_costo_privado_2030 + opt_dano_ambiental_2030
    
    dif_privado = opt_costo_privado_2030 - BAU_COSTO_PRIVADO
    dif_dano = opt_dano_ambiental_2030 - BAU_DANO_AMBIENTAL
    dif_total = opt_total - bau_total

    # --- CONSTRUCCIÓN DATAFRAMES ---
    
    # TABLA 2.1: Mitigación Política Óptima
    t2_1 = pd.DataFrame({
        'Política': ['Escenario BAU (Base)', 'Política Óptima'],
        'Emisiones CO2 [Miles Ton]': [BAU_EMISIONES_CO2/1000, opt_emisiones_co2/1000],
        'Mitigación CO2 [%]': ['', f"{mitigacion_co2_pct:.1f}%"],
        'Mitigación CO2 [Miles Ton]': ['', mitigacion_co2_ton/1000]
    })
    
    # TABLA 2.2: Costo Anual Política Óptima
    t2_2 = pd.DataFrame({
        'Política': ['Escenario BAU (Base)', 'Política Óptima', 'Diferencia'],
        'Costo (1) [US$]': [BAU_COSTO_PRIVADO, opt_costo_privado_2030, dif_privado],
        'Daño Ambiental (2) [US$]': [BAU_DANO_AMBIENTAL, opt_dano_ambiental_2030, dif_dano],
        'Total [US$]': [bau_total, opt_total, dif_total]
    })

    # Exportación
    archivo_salida = "Tablas_Tarea3_Pregunta2.xlsx"
    with pd.ExcelWriter(archivo_salida) as writer:
        t2_1.to_excel(writer, sheet_name='Tabla 2.1', index=False)
        t2_2.to_excel(writer, sheet_name='Tabla 2.2', index=False)
    
    print("\n" + "="*60)
    print(f"✅ ¡LISTO! Se ha generado el archivo '{archivo_salida}'")
    print("   Contiene las tablas 2.1 y 2.2 con el formato del PDF.")
    print("="*60)
    
    print("\nVista Previa Tabla 2.1:")
    print(t2_1)
    print("\nVista Previa Tabla 2.2:")
    print(t2_2)

else:
    print("❌ No se encontró solución óptima.")