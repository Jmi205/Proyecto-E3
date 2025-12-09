# CVRP - Capacitated Vehicle Routing Problem

## **1. Descripcion General del Proyecto**

Este proyecto implementa el **Capacitated Vehicle Routing Problem (CVRP)** con una metaheuristica y lo compara con una implementacion en pyomo:

1. **Modelo exacto con Pyomo**, utilizando programacion matematica.
2. **Metaheuristica basada en Algoritmo Genetico (GA)**, adaptado desde un ejemplo clasico de TSP pero modificado para soportar multiples vehiculos, capacidades y estructuras propias del CVRP.

El objetivo es analizar el rendimiento, calidad de soluciones, escalabilidad y comportamiento del algoritmo en tres instancias de prueba:
**caso_base, caso_2 y caso_3**.

---

## **2. Estructura del Proyecto**

```
.
├── MOS_P1_3.ipynb                          # Documento principal (informe completo)
├── resultados-pyomo-comparacion.ipynb      # Resultados exactos para cada caso con Pyomo
├── implementacion-metaheuristica-y-resultados.ipynb   # Implementacion GA + rutas + evolucion
│
├── resultados/
│   ├── pyomo/                               # 6 archivos: 3 Pyomo + 3 Metaheuristica
│   └── metaheuristica/
│
└── cvrp_content-main/                       # Archivos de entrada (clients.csv, vehicles.csv, depots.csv)
    ├── caso_base/
    ├── caso_2/
    └── caso_3/

```

### **Contenido de cada archivo principal**

* **MOS_P1_3.ipynb**
  Documento principal con descripcion del problema, analisis, comparaciones, escalabilidad y conclusiones.

* **resultados-pyomo-comparacion.ipynb**
  Scripts de Pyomo aplicados a los tres casos, sin metaheuristicas.

* **implementacion-metaheuristica-y-resultados.ipynb**
  Implementacion completa del GA, visualizacion de rutas y curva de convergencia por caso.

* **resultados**
  Resultados finales correctos y formateados para evaluacion.

* **cvrp_content-main/**
  Archivos de entrada provistos por el enunciado del proyecto.

---

## **3. Explicacion de las Modificaciones al Ejemplo GA para TSP**

El algoritmo genetico se baso en un ejemplo clasico para TSP, pero fue profundamente modificado para convertirlo en un solucionador de CVRP.
Las principales adaptaciones fueron:

### **a) Soporte para multiples vehiculos**

El TSP tradicional tiene **una sola ruta**, pero el CVRP requiere un conjunto de rutas.
Se creo un mecanismo para:

* dividir el "giant tour" en multiples rutas,
* respetar las capacidades de cada vehiculo,
* asignar un vehiculo por ruta segun su capacidad real.

### **b) Inclusion de restricciones de capacidad**

En el TSP no existen capacidades.
En esta version se computa la demanda de los clientes y se penaliza con un costo muy alto cualquier violacion.

### **c) Calculo de costos realistas**

El TSP solo usa distancia.
Aqui se incluyen:

* costo por distancia,
* costo por tiempo de recorrido,
* costo de combustible (segun eficiencia y precio),
* costo fijo por vehiculo.

### **d) Evaluacion modificada**

La funcion de fitness ya no mide solo longitud, sino costo total de operacion, lo cual cambia por completo el paisaje del problema.

### **e) Mutaciones y cruces adaptados**

Se anadieron operaciones que mantienen la viabilidad de rutas multiples, considerando swapping entre rutas y movimientos de clientes entre vehiculos.

### **f) Grafica de convergencia**

Se incorporo un registro de `best_fitness` por generacion para visualizar el comportamiento evolutivo del GA.

Estas modificaciones convierten al algoritmo en un enfoque especifico para CVRP, no simplemente una adaptacion ligera del TSP.

---

## **4. Instrucciones para Ejecutar el Codigo**

### **Requisitos Previos**

1. **Python 3.9 o superior**
2. Instalar todas las dependencias:

```bash
pip install pyomo
pip install pandas
pip install numpy
pip install matplotlib
pip install psutil
```

3. **Instalar el solver CBC**, necesario para Pyomo:

En Windows:

```bash
conda install -c conda-forge coincbc
```

En Linux / Mac:

```bash
sudo apt-get install coinor-cbc
# o
brew install cbc
```

4. Libreria para manejo de memoria (ya incluida):

```bash
pip install psutil
```

---

### **Ejecucion**

Existen dos formas de ejecutar el proyecto:

---

### **A) Desde los notebooks**

1. Abrir **resultados-pyomo-comparacion.ipynb**
   → ejecuta cada celda para obtener las soluciones exactas.

2. Abrir **implementacion-metaheuristica-y-resultados.ipynb**
   → ejecuta para obtener:

   * rutas del GA,
   * costos por vehiculo,
   * curvas de convergencia,
   * graficas de rutas.

3. Abrir **MOS_P1_3.ipynb** para ver el informe completo.

---

### **B) Ejecutar directamente el script GA (si lo usas como .py)**

```bash
python ga_cvrp.py
```

---

## **5. Dependencias y Versiones Recomendadas**

| Libreria   | Version recomendada |
| ---------- | ------------------- |
| Python     | 3.9+                |
| Pyomo      | 6.6+                |
| NumPy      | 1.26+               |
| Pandas     | 2.0+                |
| Matplotlib | 3.8+                |
| psutil     | 5.9+                |
| CBC Solver | ultima disponible   |

---

## **6. Integrantes del Grupo**

* Esteban Castelblanco Gomez – 202214942
* Juan Miguel Delgado – 202314903
* Omar Mauricio Urrego Vasquez – 202211641
* Juan Felipe Lancheros Carrillo – 202211004

