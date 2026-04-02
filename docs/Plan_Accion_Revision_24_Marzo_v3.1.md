# Plan de Acción y Protocolo de Pruebas: Revisión de la Versión 3.1 de ISAEA Nowcasting (24 de Marzo)

Este documento detalla a nivel técnico y estratégico los cambios necesarios para resolver las cuatro incidencias detectadas en la revisión del 24 de marzo de 2026. Además, se incluye un checklist de validación (pruebas) para cada punto, garantizando su resolución antes de la entrega final.

---

## 🛑 Problema 1: Error al modificar o eliminar variables y ficheros en caliente

### Descripción del Problema
Cuando el usuario, con la aplicación en marcha, elimina un archivo de entrada o deselecciona/intercambia variables de la lista de selectores, la aplicación arroja un error (típicamente de rastreo de pila en rojo, como `KeyError` o lectura de `NoneType`).

### Causa Raíz
Las interfaces de Streamlit se ejecutan de arriba a abajo con cada interacción (cambio de estado). Al borrar un fichero, Streamlit reejecuta todo el código inmediatamente. Si el código que calcula o grafica está expuesto y no tiene una validación previa de que el dato siga existiendo, se estrella.

### Cambios a Implementar
1. **Validación Condicional Temprana:** Se envolverán las llamadas de procesamiento de datos en comprobaciones condicionales robustas (`if not uploaded_file: ...`). 
2. **Uso de Mensajes Amigables (`st.warning` / `st.stop`):** En lugar de intentar ejecutar y fallar, el código interceptará la falta de inputs, detendrá la ejecución silenciosamente (`st.stop()`) y pintará en pantalla: *"Por favor, seleccione al menos una variable o introduzca un fichero válido"*.
3. **Limpieza de Caché Dinámica:** Si se quitan variables, hay que purgar de `st.session_state` los resultados anteriores para evitar cruces.

### 🧪 Protocolo de Comprobación y Chequeo
- [ ] Entrar a la app y correr el pipeline con normalidad hasta obtener resultados.
- [ ] Eliminar (clic en la 'X') el fichero maestro cargado. **Comprobación:** La pantalla debe limpiarse devolviendo el prompt inicial de "Por favor, cargue un fichero", sin tracebacks ni errores rojos.
- [ ] Intercambiar una variable del multiselect. **Comprobación:** La aplicación debe recalcular o pedir darle a ejecutar sin romperse.

### 🤖 Prompt de Ejecución
> "Por favor, ejecuta las soluciones para el **Problema 1**. Añade el control de flujo temporal y validaciones en la interfaz (Streamlit) para asegurar que la eliminación de variables o archivos no rompa la aplicación, interceptando el error con un `st.warning()` y parando la ejecución."

---

## 🛑 Problema 2: Formato incorrecto de Fechas en Excel ("update" y "fecha")

### Descripción del Problema
En los Excel descargados ("Descargas individuales" y "Generando archivo consolidado"), las columnas de actualización muestran códigos temporales internos (como `2026-10` o numeración cruda) en lugar de una fecha exacta real.

### Causa Raíz
Pandas maneja internamente fechas con el tipo `PeriodIndex` o formatos de `timestamps`. Al hacer `df.to_excel()`, este formato interno salta directamente a la exportación cruda y no resulta humanamente legible.

### Cambios a Implementar
1. **Punto de Inyección:** Localizar el controlador de exportación de DataFrames justo antes del comando `to_excel` y `to_csv`.
2. **Formateo Directo:** Aplicar una conversión explícita en las columnas conflictivas a través de: `df['fecha'] = pd.to_datetime(df['fecha']).dt.strftime('%d/%m/%Y')` (Se requerirá adaptar el parseo para establecer el primer o el último día del mes en curso).

### 🧪 Protocolo de Comprobación y Chequeo
- [ ] Ejecutar el pipeline de nowcasting al completo.
- [ ] Descargar o abrir el `nowcasting_individual.xlsx` general.
- [ ] **Comprobación:** Buscar la columna `fecha` y `update`. Confirmar que los valores están como texto/fecha clásicos en formato `DD/MM/YYYY` (ejemplo: `31/10/2026`) en un 100% de las filas.
- [ ] Repetir la comprobación exacta sobre el `consolidado_impactos.xlsx`.

### 🤖 Prompt de Ejecución
> "Por favor, ejecuta las soluciones para el **Problema 2**. Modifica el código de renderizado y exportación para asegurar que las columnas *update* y *fecha* salgan como fechas formateadas `DD/MM/YYYY` antes de guardar en Excel y no como periodos o timestamps."

---

## 🛑 Problema 3: Desfase en los valores del gráfico de impactos ("salto estimación")

### Descripción del Problema
El gráfico captura bien cuál es el "impacto final" (el aumento total, por ejemplo, 0.12), pero las barras o líneas no lo ubican visualmente entre su estimación de origen ("punto inicial" = 0.34) y su estimación de destino ("punto de llegada" = 0.45).

### Causa Raíz
La gráfica actual no se está anclando a la "estimación anterior" como base (*baseline*). Posiblemente hace un gráfico acumulativo estándar partiendo desde 0 o centra las anomalías.

### Cambios a Implementar
1. **Refactorización de la Serie Visual:** En la función encargada de generar el gráfico (ya sea `matplotlib`, `plotly` u otra), hay que configurar explícitamente el origen Y de la barra inicial.
2. Si es un *waterfall plot* o barras acumuladas, el punto de partida (la barra base transparente o elemento inicial) equivaldrá a `estimación_previa`. 
3. El techo absoluto de cada impacto sumado debe ir alineado sobre el eje Y total.

### 🧪 Protocolo de Comprobación y Chequeo
- [ ] Ejecutar y situar la atención en el gráfico principal de impacto.
- [ ] Mirar la tabla de resultados para un momento T. Si el valor en **T-1 es 0.34**, y el **nuevo en T es 0.45**...
- [ ] **Comprobación Visual:** El gráfico debe empezar explícitamente en el nivel 0.34 del Y-axis. 
- [ ] **Comprobación Visual:** El escalón resultante final de la gráfica debe estar a nivel 0.45, correspondiéndose de forma precisa con la aritmética: `0.34 + impacto = 0.45`.

### 🤖 Prompt de Ejecución
> "Por favor, ejecuta las soluciones para el **Problema 3**. Refactoriza el gráfico de impactos (waterfall/barras) para que el punto inicial del gráfico se corresponda con la *estimación previa* (origen Y) y el nivel máximo/final coincida matemáticamente con la *nueva estimación*."

---

## 🛑 Problema 4: La caché anula los cambios manuales forzados en los Excel de origen

### Descripción del Problema
Cuando se altera deliberada y masivamente un dato empírico en un Excel que ya está cargado previamente (ej. cambiar las Cifras de Negocio base de 162 a 325), la aplicación sigue devolviendo el mismo output en el consolidado.

### Causa Raíz
Mecanismo de Memoización (*caching*) ineficiente por parte de `@st.cache_data`. Streamlit solo asocia que el archivo se llama "Envío_21_02_2026.xlsx". Cuando lo vuelves a arrastrar, verifica que "es el mismo archivo" por su nombre y te devuelve los datos copiados en memoria central, saltándose el análisis de tus nuevos datos alterados.

### Cambios a Implementar
1. **Invalidación Dura en el Hashing de Caché:** Añadir un parámetro adicional a las variables *state* o, si se cargan por upload, leer el hash del contenido en bytes para que Streamlit detecte que es internamente diferente.
2. **Botón Físico de "Limpieza Total":** Añadir en el menú lateral o inferior de Streamlit un botón de seguridad: `st.button("Limpiar Caché y Forzar Recálculo")`, que dispare internamente la llamada segura a `st.cache_data.clear()` junto a `st.session_state.clear()`.

### 🧪 Protocolo de Comprobación y Chequeo
- [ ] Cargar el Excel estándar y estimar. Anotar el impacto o el valor (ej. consolidado en X periodo dice `0.11`).
- [ ] Abrir el Excel, forzar un error catastrófico intencionado (ej. cambiar 117 a 99999). Guardar localmente.
- [ ] Cerrar y re-cargar el archivo en Streamlit (o presionar el botón "Limpiar Caché" y re-subir).
- [ ] **Comprobación:** El archivo consolidado final **debe** presentar resultados totalmente distintos, certificando que la aplicación ha devorado las cifras frescas y ha ignorado su memoria previa.

### 🤖 Prompt de Ejecución
> "Por favor, ejecuta las soluciones para el **Problema 4**. Invalida correctamente la caché de Streamlit incorporando un sistema que lea el hash/fecha de modificación del archivo, e implementa el botón de *Limpiar Caché y Recalcular* en la interfaz para forzar actualizaciones directas en el Excel de origen."
