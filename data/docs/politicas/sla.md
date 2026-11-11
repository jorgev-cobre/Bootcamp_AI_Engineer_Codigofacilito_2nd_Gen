# Niveles de servicio (SLA)

Esta política define en cuánto tiempo responde y resuelve Acme Operaciones cada ticket según su prioridad, y cómo se miden los tiempos de los proveedores.

## SLA internos por prioridad

### Prioridad alta

- Primera respuesta: 15 minutos, las 24 horas.
- Resolución o plan alternativo: 8 horas.
- Seguimiento al proveedor: cada hora.

### Prioridad media

- Primera respuesta: 2 horas hábiles.
- Resolución: 2 días hábiles.

### Prioridad baja

- Primera respuesta: 1 día hábil.
- Resolución: 5 días hábiles.

## Horario hábil

El horario hábil es de lunes a viernes de 9:00 a 18:00, hora del centro de México. Los días festivos oficiales no son hábiles. Los tickets de prioridad alta no usan horario hábil: su reloj corre siempre.

## SLA de proveedores

Cada proveedor tiene su propio tiempo de respuesta, definido en su contrato de servicio. Los tiempos vigentes para los proveedores con contrato en este repositorio son:

- PagosRed (pasarela de pagos): 4 horas hábiles.
- NubeHost (hospedaje del portal y del sitio): 2 horas hábiles.
- Transportes Sierra (fletes): 12 horas hábiles.

El tiempo de respuesta del proveedor se mide desde que se abre el caso con él hasta su primera respuesta con diagnóstico. Una respuesta automática no cuenta.

## Medición y reporte

Finanzas publica cada mes el porcentaje de tickets resueltos dentro del SLA por equipo. Un equipo por debajo de 90% durante dos meses seguidos revisa su proceso con la Guardia N3.
