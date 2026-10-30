# Runbook: portal de proveedores

El portal de proveedores es donde los proveedores suben facturas, consultan pagos y responden cotizaciones. Lo hospeda NubeHost, que también hospeda el sitio web. Tickets de referencia: T-1043 (portal de proveedores caído, error 503), T-1016 (página de cotizaciones muy lenta) y T-1005 (certificado del sitio por vencer).

## Síntomas

- El portal no carga o responde con error 503 al subir facturas, como en el T-1043.
- La sección de cotizaciones tarda más de 10 segundos en cargar.
- El navegador advierte que el certificado del sitio no es válido o está por vencer.

## Diagnóstico

1. Abre el portal desde la red de la oficina y desde una red móvil. Si falla en las dos, el problema no es de nuestra red.
2. Revisa el panel de NubeHost: uso de CPU, memoria y estado del balanceador.
3. Si el error es 503, revisa si NubeHost tiene una ventana de mantenimiento programada.
4. Si el problema es lentitud, compara el tiempo de carga antes y después del último cambio de servidor; en el T-1016 la lentitud empezó con una migración.

## Recuperación

### Portal caído

1. Abre un ticket con NubeHost en mesa@nubehost.example con la hora de inicio y capturas del error.
2. Publica un aviso en la página de inicio del sitio: "El portal de proveedores no está disponible; las facturas pueden enviarse por correo a cuentas.por.pagar@acme.example".
3. Pide a NubeHost reiniciar el servicio del portal si no hay mantenimiento en curso.

### Certificado por vencer

1. Revisa la fecha de vencimiento del certificado en el panel de NubeHost.
2. Solicita la renovación al menos 15 días antes del vencimiento.
3. Verifica el nuevo certificado desde tres navegadores distintos.

## A quién avisar

- Guardia N1 de TI desde el primer síntoma.
- Compras y Cuentas por Pagar si el portal está caído más de una hora en día de cierre.
- Si el ticket es de prioridad alta y NubeHost no responde a tiempo, aplica la política de escalamiento.
