# Proyecto ISAEA - IECA

Ajuste 28 de marzo 2025 (versiones search 2.1 y app 2.3)

- Introducción de una nueva caja donde el usuario puede incluir series que quiere que siempre estén en las especificaciones evaluadas. Esta opción son tanto para series mensuales como trimestrales
- Se añade barra lateral donde incluir path del directorio de salida que se desea para el fichero
- La introducción de dummies para el COVID son ahora True por defecto.
- Nuevo gráfico de PIB

## Actualización Abril 2026 (Versión V3.3)

**Resolución de Bugs, Estabilización e IA Analítica**
- **Soporte Dinámico para Cambios en Caliente (Hot-Swapping):** Implementados cortafuegos y un re-cálculo autoconmutado cada vez que el usuario altera los check-box de los predictores en la barra lateral lateral o elimina los documentos fuente, evadiendo así roturas críticas en la ejecución.
- **Normalización de Fechas para Entregables Excel:** Centralizado un conversor de fechas avanzado. Se elimina toda opacidad del renderizado de Pandas/Statsmodels (ej. '2026Q3' o '2026-10') logrando que la exportación *Nowcasting consolidado y parcial* luzca permanentemente en un legible `DD/MM/YYYY`. Garantiza un análisis sin roces estructurales.
- **Gráfico Universal de Impactos (Waterfall Parametrizado):** Unificación de la matemática visual de estimación suprimiendo la necesidad engañosa de un doble eje. Los bloques estéticos que explican las variaciones nacen ahora estrictamente anclados frente al pivote de la ola predictiva anterior (T-1).
- **Hard-reset para Manipuladores de Excel (Control MD5):** Las subidas manuales constantes con mismo nombre se interpretan ahora perfectamente gracias al sistema de criptografía MD5 incorporado internamente y a la inyección física de un Botón de Limpieza y Purgamiento Directo en la consola lateral (`🔄 Limpiar Caché y Recalcular`).
