# Runbook: caída de la pasarela de pagos

Este runbook aplica cuando falla el cobro con tarjeta que procesa PagosRed, nuestro proveedor de pasarela de pagos. Lo usa la guardia de Finanzas y TI. Tickets de referencia: T-1042 (factura duplicada de octubre, doble cargo al cliente), T-1003 (cobro con tarjeta rechazado en la tienda en línea) y T-1014 (conciliación de pagos con diferencias).

## Síntomas

- Los clientes reportan que el pago con tarjeta se rechaza al final del carrito aunque tengan saldo.
- Aparecen cobros duplicados: el cliente recibe dos cargos o dos facturas con el mismo folio, como en el T-1042.
- La conciliación bancaria muestra cobros que no aparecen en el estado de cuenta.
- El tablero de pagos marca más de 5% de transacciones fallidas en 15 minutos.

## Diagnóstico inicial

1. Revisa la página de estado de PagosRed y el correo de soporte@pagosred.example. Si PagosRed ya declaró un incidente, anota su número en el ticket.
2. Compara el número de transacciones aprobadas de la última hora contra el mismo horario de la semana anterior.
3. Busca en el ticket el folio de la factura y los últimos cuatro dígitos de la tarjeta, nunca la tarjeta completa.
4. Confirma si el problema es solo de PagosRed o también de nuestra tienda en línea: si la tienda no carga, sigue el runbook del portal y no este.

## Contención

### Si hay cobros duplicados

1. No canceles cobros desde nuestro sistema: la cancelación la hace PagosRed para que la conciliación cuadre.
2. Abre un caso con PagosRed para revertir el cargo y anota el número de caso en el ticket. En el T-1042, Finanzas confirmó el doble cargo y abrió el caso con la pasarela dos días después; el objetivo es abrirlo el mismo día.
3. Cancela la factura duplicada en el sistema de facturación y emite la nota de crédito si el cliente ya la timbró.
4. Avisa al cliente que el reembolso depende del proceso de PagosRed y dale el número de caso.

### Si los cobros se rechazan

1. Activa el aviso de la tienda: "Estamos teniendo problemas con el pago con tarjeta; puedes pagar por transferencia".
2. Habilita el pago por transferencia como método temporal.
3. Reporta el incidente a PagosRed con la hora de inicio, el porcentaje de rechazos y tres folios de ejemplo.

## A quién avisar

- Guardia N1 de TI desde el primer síntoma.
- Dueño del servicio de pagos (Finanzas) si el incidente dura más de 30 minutos.
- Si el ticket es de prioridad alta y PagosRed no responde a tiempo, aplica la política de escalamiento.

## Cierre del incidente

1. Confirma con PagosRed que el incidente está resuelto y que los reembolsos quedaron aplicados.
2. Corre la conciliación del día y verifica que no queden diferencias, como las del T-1014.
3. Deja en el ticket la causa, la duración y las acciones; marca el ticket como resuelto solo cuando el cliente confirme el reembolso.
