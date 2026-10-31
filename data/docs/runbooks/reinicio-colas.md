# Runbook: reinicio de colas de procesamiento

Las colas de procesamiento mueven pedidos del ERP al almacén y facturas al sistema contable. Cuando una cola se atora, los pedidos no llegan al almacén y se acumulan capturas dobles, como las que reporta el T-1022 (doble captura de pedidos).

## Síntomas

- El tablero de colas muestra más de 500 mensajes pendientes en la cola de pedidos.
- El almacén no recibe pedidos nuevos desde hace más de 20 minutos.
- Ventas reporta que los pedidos aparecen en el ERP pero no en el almacén.

## Antes de reiniciar

1. Confirma en el tablero cuál cola está atorada: pedidos, facturas o notificaciones.
2. Revisa si hay un proceso de cierre contable en curso. Durante el cierre no se reinicia la cola de facturas sin autorización de Finanzas.
3. Exporta la lista de mensajes pendientes y guárdala en el ticket.

## Reinicio

### Cola de pedidos

1. Detén el consumidor de la cola de pedidos desde la consola de operaciones.
2. Espera a que el contador de mensajes en proceso llegue a cero.
3. Inicia el consumidor de nuevo y verifica que el contador de pendientes baje en los siguientes cinco minutos.

### Cola de facturas

1. Pide autorización a Finanzas si es día de cierre.
2. Detén el consumidor, limpia los mensajes marcados como inválidos y vuelve a iniciarlo.
3. Revisa que las facturas del día aparezcan en el sistema contable.

## Después del reinicio

1. Verifica con el almacén que los pedidos atrasados ya llegaron.
2. Si un pedido se procesó dos veces, cancélalo en el almacén, no en el ERP.
3. Registra en el ticket la hora del reinicio, la cola y cuántos mensajes se reprocesaron.

## A quién avisar

- Guardia N1 de TI antes de reiniciar.
- Logística si la cola de pedidos estuvo detenida más de una hora.
