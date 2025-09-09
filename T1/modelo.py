from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Definir el modelo de optimización
modelo = LpProblem("Optimización_Sistema_Electrico", LpMinimize)

# Definir las tecnologías y bloques de demanda
tecnologias_existentes = ['Carbón', 'Biomasa', 'Petróleo', 'CC-GNL']
tecnologias_nuevas = ['Solar FV', 'Eólica', 'Geotérmica']
bloques_demanda = ['D1', 'D2', 'D3']

# Datos (deben ser extraídos de los archivos de datos)
demanda = {'D1': 200, 'D2': 300, 'D3': 250}  # Demanda por bloque (en MW)
capacidad = {'Carbón': 1.0, 'Biomasa': 0.8, 'Petróleo': 0.5, 'CC-GNL': 1.5}  # Ejemplo de capacidad de las tecnologías (en MW)
disponibilidad = {'Carbón': 0.88, 'Biomasa': 0.9, 'Petróleo': 0.9, 'CC-GNL': 0.93}  # Factor de disponibilidad

# Costos de operación (en $/MWh)
costo_variable = {'Carbón': 1.9, 'Biomasa': 2.5, 'Petróleo': 11.5, 'CC-GNL': 3.4}

# Costos de inversión de las nuevas tecnologías (en $/kW)
costo_inversion = {'Solar FV': 700, 'Eólica': 900, 'Geotérmica': 1200}

# Capacidad máxima de instalación (en MW) para nuevas tecnologías
limite_instalacion = {'Solar FV': 500, 'Eólica': 600, 'Geotérmica': 400}

# Variables de decisión
E = {}  # Energía generada por cada tecnología en cada bloque
PN = {}  # Potencia instalada de tecnologías nuevas
EN = {}  # Energía generada por tecnologías nuevas

# Crear las variables de decisión para las energías generadas
for t in tecnologias_existentes:
    for d in bloques_demanda:
        E[t, d] = LpVariable(f"E_{t}_{d}", lowBound=0)

for t in tecnologias_nuevas:
    for d in bloques_demanda:
        EN[t, d] = LpVariable(f"EN_{t}_{d}", lowBound=0)
    PN[t] = LpVariable(f"PN_{t}", lowBound=0)  # Potencia instalada de nuevas tecnologías

# Función objetivo - Caso Base (2016)
modelo += lpSum([costo_variable[t] * E[t, d] for t in tecnologias_existentes for d in bloques_demanda])

# Función objetivo - Escenario 2030 (con nuevas tecnologías)
modelo_2030 = LpProblem("Escenario_2030", LpMinimize)
modelo_2030 += lpSum([costo_variable[t] * E[t, d] for t in tecnologias_existentes for d in bloques_demanda])
modelo_2030 += lpSum([costo_inversion[t] * PN[t] for t in tecnologias_nuevas])
modelo_2030 += lpSum([costo_variable[t] * EN[t, d] for t in tecnologias_nuevas for d in bloques_demanda])

# Función objetivo - Política ERNC (2030)
modelo_ernc = LpProblem("Política_ERNC_2030", LpMinimize)
modelo_ernc += modelo_2030.objective  # Tomar la misma función objetivo que el escenario 2030
# Restricción para que al menos el 30% de la generación provenga de ERNC
modelo_ernc += lpSum([EN[t, d] for t in ['Solar FV', 'Eólica', 'Geotérmica'] for d in bloques_demanda]) >= 0.3 * lpSum([E[t, d] for t in tecnologias_existentes for d in bloques_demanda])

# Restricciones para el balance de demanda
for d in bloques_demanda:
    modelo_ernc += lpSum([E[t, d] for t in tecnologias_existentes]) + lpSum([EN[t, d] for t in tecnologias_nuevas]) >= demanda[d], f"Balance_demanda_{d}"

# Restricciones para las capacidades
for t in tecnologias_existentes:
    for d in bloques_demanda:
        modelo_ernc += E[t, d] <= capacidad[t] * disponibilidad[t], f"Capacidad_{t}_{d}"

# Restricciones de instalación para las nuevas tecnologías
for t in tecnologias_nuevas:
    modelo_ernc += PN[t] <= limite_instalacion[t], f"Limitacion_instalacion_{t}"

# Resolver el modelo
modelo_ernc.solve()
