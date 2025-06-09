"""
Spanish conversation samples for testing the LLM Analyzer.
These simulate realistic Whisper transcription outputs.
"""

from datetime import datetime
from typing import List, Dict

# Sample conversations with different scenarios
CONVERSATIONS = {
    "business_meeting": {
        "id": "conv-001",
        "title": "Reunión de Planificación Trimestral",
        "raw_transcript": """María García: Buenos días a todos. Vamos a comenzar con la reunión de planificación del trimestre.
Carlos Rodríguez: Buenos días María. Tengo preparado el informe de ventas del último mes.
María García: Perfecto Carlos. Antes de eso, quiero que revisemos los objetivos pendientes.
Ana López: Disculpen el retraso. Tuve un problema con el tráfico.
María García: No te preocupes Ana. Carlos, por favor comparte tu pantalla con las cifras.
Carlos Rodríguez: Claro. Como pueden ver, las ventas aumentaron un 15% respecto al mes anterior.
Ana López: Excelente resultado. ¿Cuáles fueron los productos estrella?
Carlos Rodríguez: Los productos de la línea premium tuvieron mejor desempeño.
María García: Necesitamos mantener ese momentum. Carlos, prepara un análisis detallado para el viernes.
Carlos Rodríguez: Lo tendré listo. ¿Incluyo las proyecciones para el siguiente trimestre?
María García: Sí, por favor. Ana, necesito que coordines con marketing una campaña para los productos premium.
Ana López: Me encargo. Voy a agendar una reunión con ellos esta tarde.
María García: Perfecto. Hay un tema importante: el presupuesto para el próximo trimestre.
Carlos Rodríguez: Tengo una propuesta. Podríamos aumentar la inversión en marketing digital.
Ana López: Apoyo la idea. Hemos visto buenos resultados con las campañas online.
María García: Preparen una propuesta formal. La necesito para la junta directiva del lunes.
Carlos Rodríguez: ¿Trabajamos juntos en eso Ana?
Ana López: Por supuesto. Podemos reunirnos mañana por la mañana.
María García: Excelente. No olviden incluir el ROI proyectado.
Carlos Rodríguez: Entendido. También sugiero que revisemos los costos operativos.
María García: Buen punto. Agéndalo como tema para la próxima reunión.
Ana López: María, ¿qué hay del proyecto de expansión que mencionaste la semana pasada?
María García: Sigue en evaluación. Necesitamos los números del trimestre completo.
Carlos Rodríguez: Puedo adelantar algunas proyecciones si ayuda.
María García: Sería útil. Envíamelas por correo cuando las tengas.
Ana López: Hay que considerar también la capacitación del personal nuevo.
María García: Cierto. Ana, añade eso a tu lista de pendientes.
Ana López: Anotado. ¿Cuándo sería la fecha tentativa de contratación?
María García: Si todo sale bien, iniciaríamos en abril.
Carlos Rodríguez: Perfecto. Eso nos da tiempo para preparar el onboarding.
María García: Exacto. Bueno, creo que cubrimos los puntos principales. ¿Alguna pregunta?
Ana López: Solo confirmar: mi entregable es la propuesta de marketing para el lunes, ¿correcto?
María García: Correcto, junto con Carlos. Carlos, tú además el análisis detallado para el viernes.
Carlos Rodríguez: Entendido. Lo tendré listo.
María García: Perfecto. Nos vemos el viernes entonces. Que tengan buen día.""",
        "speaker_transcript": """[Hablante 1] Buenos días a todos. Vamos a comenzar con la reunión de planificación del trimestre.
[Hablante 2] Buenos días María. Tengo preparado el informe de ventas del último mes.
[Hablante 1] Perfecto Carlos. Antes de eso, quiero que revisemos los objetivos pendientes.
[Hablante 3] Disculpen el retraso. Tuve un problema con el tráfico.
[Hablante 1] No te preocupes Ana. Carlos, por favor comparte tu pantalla con las cifras.
[Hablante 2] Claro. Como pueden ver, las ventas aumentaron un 15% respecto al mes anterior.
[Hablante 3] Excelente resultado. ¿Cuáles fueron los productos estrella?
[Hablante 2] Los productos de la línea premium tuvieron mejor desempeño.
[Hablante 1] Necesitamos mantener ese momentum. Carlos, prepara un análisis detallado para el viernes.
[Hablante 2] Lo tendré listo. ¿Incluyo las proyecciones para el siguiente trimestre?
[Hablante 1] Sí, por favor. Ana, necesito que coordines con marketing una campaña para los productos premium.
[Hablante 3] Me encargo. Voy a agendar una reunión con ellos esta tarde.
[Hablante 1] Perfecto. Hay un tema importante: el presupuesto para el próximo trimestre.
[Hablante 2] Tengo una propuesta. Podríamos aumentar la inversión en marketing digital.
[Hablante 3] Apoyo la idea. Hemos visto buenos resultados con las campañas online.
[Hablante 1] Preparen una propuesta formal. La necesito para la junta directiva del lunes.
[Hablante 2] ¿Trabajamos juntos en eso Ana?
[Hablante 3] Por supuesto. Podemos reunirnos mañana por la mañana.
[Hablante 1] Excelente. No olviden incluir el ROI proyectado.
[Hablante 2] Entendido. También sugiero que revisemos los costos operativos.
[Hablante 1] Buen punto. Agéndalo como tema para la próxima reunión.
[Hablante 3] María, ¿qué hay del proyecto de expansión que mencionaste la semana pasada?
[Hablante 1] Sigue en evaluación. Necesitamos los números del trimestre completo.
[Hablante 2] Puedo adelantar algunas proyecciones si ayuda.
[Hablante 1] Sería útil. Envíamelas por correo cuando las tengas.
[Hablante 3] Hay que considerar también la capacitación del personal nuevo.
[Hablante 1] Cierto. Ana, añade eso a tu lista de pendientes.
[Hablante 3] Anotado. ¿Cuándo sería la fecha tentativa de contratación?
[Hablante 1] Si todo sale bien, iniciaríamos en abril.
[Hablante 2] Perfecto. Eso nos da tiempo para preparar el onboarding.
[Hablante 1] Exacto. Bueno, creo que cubrimos los puntos principales. ¿Alguna pregunta?
[Hablante 3] Solo confirmar: mi entregable es la propuesta de marketing para el lunes, ¿correcto?
[Hablante 1] Correcto, junto con Carlos. Carlos, tú además el análisis detallado para el viernes.
[Hablante 2] Entendido. Lo tendré listo.
[Hablante 1] Perfecto. Nos vemos el viernes entonces. Que tengan buen día.""",
        "duration": 900,  # 15 minutes
        "speakers": ["María García", "Carlos Rodríguez", "Ana López"],
        "expected_tasks": [
            {
                "description": "Preparar análisis detallado de ventas",
                "assignee": "Carlos",
                "due_date": "viernes",
                "assigned_by": "María"
            },
            {
                "description": "Coordinar campaña de marketing para productos premium",
                "assignee": "Ana",
                "assigned_by": "María"
            },
            {
                "description": "Preparar propuesta formal de presupuesto con ROI",
                "assignee": "Carlos y Ana",
                "due_date": "lunes",
                "assigned_by": "María"
            }
        ]
    },
    
    "tech_support": {
        "id": "conv-002",
        "title": "Soporte Técnico - Problema con Sistema",
        "raw_transcript": """Juan Méndez: Hola, buenos días. Tengo un problema con el sistema de inventario.
Luis Fernández: Buenos días Juan. Soy Luis del soporte técnico. ¿Qué problema estás experimentando?
Juan Méndez: El sistema se congela cuando intento generar el reporte mensual.
Luis Fernández: ¿Desde cuándo está pasando esto?
Juan Méndez: Empezó ayer por la tarde. Antes funcionaba perfectamente.
Luis Fernández: Entiendo. ¿Has intentado reiniciar el sistema?
Juan Méndez: Sí, lo reinicié varias veces pero el problema persiste.
Luis Fernández: Voy a necesitar revisar los logs del sistema. ¿Puedes darme acceso remoto?
Juan Méndez: Claro, te envío las credenciales por el chat interno.
Luis Fernández: Perfecto. Dame un momento mientras me conecto... Ya estoy dentro.
Juan Méndez: ¿Ves algo raro?
Luis Fernández: Sí, hay un error de memoria. Parece que la base de datos creció mucho.
Juan Méndez: ¿Eso es grave? Necesito el reporte para hoy.
Luis Fernández: No te preocupes. Voy a optimizar la base de datos ahora mismo.
Juan Méndez: ¿Cuánto tiempo tomará?
Luis Fernández: Unos 20 minutos aproximadamente. Te aviso cuando termine.
Juan Méndez: Okay. ¿Hay algo que deba hacer mientras tanto?
Luis Fernández: No uses el sistema por ahora. Ah, y para el futuro, hay que programar mantenimientos regulares.
Juan Méndez: No sabía que era necesario. ¿Cada cuánto tiempo?
Luis Fernández: Recomiendo una vez al mes. Puedo configurar una tarea automática si quieres.
Juan Méndez: Sería excelente. No quiero que vuelva a pasar.
Luis Fernández: Listo, ya terminé la optimización. Prueba ahora a generar el reporte.
Juan Méndez: Déjame ver... ¡Funciona perfectamente! Muchas gracias Luis.
Luis Fernández: De nada. Voy a documentar este caso y programar el mantenimiento mensual.
Juan Méndez: Perfecto. Una última cosa, ¿podrías enviarme las instrucciones por si pasa de nuevo?
Luis Fernández: Claro, te envío un documento con los pasos básicos de troubleshooting.
Juan Méndez: Genial. Gracias por la ayuda rápida.
Luis Fernández: Para eso estamos. No dudes en contactarnos si tienes más problemas.""",
        "speaker_transcript": """[Hablante 1] Hola, buenos días. Tengo un problema con el sistema de inventario.
[Hablante 2] Buenos días Juan. Soy Luis del soporte técnico. ¿Qué problema estás experimentando?
[Hablante 1] El sistema se congela cuando intento generar el reporte mensual.
[Hablante 2] ¿Desde cuándo está pasando esto?
[Hablante 1] Empezó ayer por la tarde. Antes funcionaba perfectamente.
[Hablante 2] Entiendo. ¿Has intentado reiniciar el sistema?
[Hablante 1] Sí, lo reinicié varias veces pero el problema persiste.
[Hablante 2] Voy a necesitar revisar los logs del sistema. ¿Puedes darme acceso remoto?
[Hablante 1] Claro, te envío las credenciales por el chat interno.
[Hablante 2] Perfecto. Dame un momento mientras me conecto... Ya estoy dentro.
[Hablante 1] ¿Ves algo raro?
[Hablante 2] Sí, hay un error de memoria. Parece que la base de datos creció mucho.
[Hablante 1] ¿Eso es grave? Necesito el reporte para hoy.
[Hablante 2] No te preocupes. Voy a optimizar la base de datos ahora mismo.
[Hablante 1] ¿Cuánto tiempo tomará?
[Hablante 2] Unos 20 minutos aproximadamente. Te aviso cuando termine.
[Hablante 1] Okay. ¿Hay algo que deba hacer mientras tanto?
[Hablante 2] No uses el sistema por ahora. Ah, y para el futuro, hay que programar mantenimientos regulares.
[Hablante 1] No sabía que era necesario. ¿Cada cuánto tiempo?
[Hablante 2] Recomiendo una vez al mes. Puedo configurar una tarea automática si quieres.
[Hablante 1] Sería excelente. No quiero que vuelva a pasar.
[Hablante 2] Listo, ya terminé la optimización. Prueba ahora a generar el reporte.
[Hablante 1] Déjame ver... ¡Funciona perfectamente! Muchas gracias Luis.
[Hablante 2] De nada. Voy a documentar este caso y programar el mantenimiento mensual.
[Hablante 1] Perfecto. Una última cosa, ¿podrías enviarme las instrucciones por si pasa de nuevo?
[Hablante 2] Claro, te envío un documento con los pasos básicos de troubleshooting.
[Hablante 1] Genial. Gracias por la ayuda rápida.
[Hablante 2] Para eso estamos. No dudes en contactarnos si tienes más problemas.""",
        "duration": 600,  # 10 minutes
        "speakers": ["Juan Méndez", "Luis Fernández"],
        "expected_tasks": [
            {
                "description": "Documentar el caso de error de memoria",
                "assignee": "Luis",
                "assigned_by": "Luis"
            },
            {
                "description": "Programar mantenimiento mensual automático",
                "assignee": "Luis",
                "assigned_by": "Luis"
            },
            {
                "description": "Enviar instrucciones de troubleshooting",
                "assignee": "Luis",
                "assigned_by": "Juan"
            }
        ]
    },
    
    "client_call": {
        "id": "conv-003",
        "title": "Llamada con Cliente - Cotización",
        "raw_transcript": """Sandra Ruiz: Buenas tardes, habla Sandra de ventas. ¿En qué puedo ayudarte?
Pedro Jiménez: Hola Sandra, soy Pedro Jiménez de Construcciones del Norte. Necesito una cotización.
Sandra Ruiz: Con gusto Pedro. ¿Qué materiales necesitas?
Pedro Jiménez: Necesito 500 metros de cable calibre 12 y 200 cajas de registro.
Sandra Ruiz: Perfecto. ¿Para cuándo los necesitarías?
Pedro Jiménez: Es urgente, los necesito para el lunes si es posible.
Sandra Ruiz: Déjame verificar el inventario... Sí, tenemos disponibilidad.
Pedro Jiménez: Excelente. ¿Cuál sería el precio?
Sandra Ruiz: El cable está a 45 pesos el metro y las cajas a 120 pesos cada una.
Pedro Jiménez: ¿Hay algún descuento por volumen?
Sandra Ruiz: Por la cantidad que solicitas, puedo ofrecerte un 8% de descuento.
Pedro Jiménez: Me parece bien. ¿Incluye el envío?
Sandra Ruiz: El envío tiene un costo adicional de 500 pesos. ¿A qué dirección sería?
Pedro Jiménez: A nuestra bodega en el parque industrial, en la calle 5 norte.
Sandra Ruiz: Conozco la zona. Podemos entregar el lunes por la mañana.
Pedro Jiménez: Perfecto. ¿Me puedes enviar la cotización formal por correo?
Sandra Ruiz: Por supuesto. ¿A qué correo la envío?
Pedro Jiménez: A pjimenez@construccionesdelnorte.com
Sandra Ruiz: Anotado. La tendrás en menos de una hora.
Pedro Jiménez: Una cosa más, ¿manejan crédito?
Sandra Ruiz: Sí, manejamos crédito a 30 días para clientes frecuentes.
Pedro Jiménez: Nosotros hemos comprado varias veces con ustedes.
Sandra Ruiz: Déjame verificar... Sí, ya tienen línea de crédito aprobada.
Pedro Jiménez: Excelente. Entonces procedemos con el pedido.
Sandra Ruiz: Perfecto. Te envío la cotización y si estás de acuerdo, la conviertes en orden de compra.
Pedro Jiménez: Muy bien. Esperaré tu correo.
Sandra Ruiz: ¿Necesitas algo más Pedro?
Pedro Jiménez: Por ahora es todo. Muchas gracias Sandra.
Sandra Ruiz: A ti por tu preferencia. Que tengas buena tarde.""",
        "speaker_transcript": """[Hablante 1] Buenas tardes, habla Sandra de ventas. ¿En qué puedo ayudarte?
[Hablante 2] Hola Sandra, soy Pedro Jiménez de Construcciones del Norte. Necesito una cotización.
[Hablante 1] Con gusto Pedro. ¿Qué materiales necesitas?
[Hablante 2] Necesito 500 metros de cable calibre 12 y 200 cajas de registro.
[Hablante 1] Perfecto. ¿Para cuándo los necesitarías?
[Hablante 2] Es urgente, los necesito para el lunes si es posible.
[Hablante 1] Déjame verificar el inventario... Sí, tenemos disponibilidad.
[Hablante 2] Excelente. ¿Cuál sería el precio?
[Hablante 1] El cable está a 45 pesos el metro y las cajas a 120 pesos cada una.
[Hablante 2] ¿Hay algún descuento por volumen?
[Hablante 1] Por la cantidad que solicitas, puedo ofrecerte un 8% de descuento.
[Hablante 2] Me parece bien. ¿Incluye el envío?
[Hablante 1] El envío tiene un costo adicional de 500 pesos. ¿A qué dirección sería?
[Hablante 2] A nuestra bodega en el parque industrial, en la calle 5 norte.
[Hablante 1] Conozco la zona. Podemos entregar el lunes por la mañana.
[Hablante 2] Perfecto. ¿Me puedes enviar la cotización formal por correo?
[Hablante 1] Por supuesto. ¿A qué correo la envío?
[Hablante 2] A pjimenez@construccionesdelnorte.com
[Hablante 1] Anotado. La tendrás en menos de una hora.
[Hablante 2] Una cosa más, ¿manejan crédito?
[Hablante 1] Sí, manejamos crédito a 30 días para clientes frecuentes.
[Hablante 2] Nosotros hemos comprado varias veces con ustedes.
[Hablante 1] Déjame verificar... Sí, ya tienen línea de crédito aprobada.
[Hablante 2] Excelente. Entonces procedemos con el pedido.
[Hablante 1] Perfecto. Te envío la cotización y si estás de acuerdo, la conviertes en orden de compra.
[Hablante 2] Muy bien. Esperaré tu correo.
[Hablante 1] ¿Necesitas algo más Pedro?
[Hablante 2] Por ahora es todo. Muchas gracias Sandra.
[Hablante 1] A ti por tu preferencia. Que tengas buena tarde.""",
        "duration": 480,  # 8 minutes
        "speakers": ["Sandra Ruiz", "Pedro Jiménez"],
        "expected_tasks": [
            {
                "description": "Enviar cotización formal por correo",
                "assignee": "Sandra",
                "due_date": "en menos de una hora",
                "assigned_by": "Pedro"
            },
            {
                "description": "Entregar pedido (500m cable y 200 cajas)",
                "assignee": "Sandra/Empresa",
                "due_date": "lunes por la mañana",
                "assigned_by": "Pedro"
            }
        ]
    },
    
    "team_standup": {
        "id": "conv-004",
        "title": "Daily Standup - Equipo de Desarrollo",
        "raw_transcript": """Roberto Silva: Buenos días equipo. Empecemos con el daily. Elena, ¿cómo vas?
Elena Martín: Buenos días. Ayer terminé la integración con la API de pagos. Hoy voy a trabajar en las pruebas unitarias.
Roberto Silva: Genial. ¿Algún bloqueador?
Elena Martín: No, todo bien por ahora.
Roberto Silva: Perfecto. Diego, tu turno.
Diego Castro: Hola. Estoy atascado con el bug del módulo de reportes. Necesito ayuda.
Roberto Silva: ¿Qué tipo de ayuda necesitas?
Diego Castro: No encuentro dónde está el error. Ya revisé el código tres veces.
Elena Martín: Yo puedo ayudarte después del standup.
Diego Castro: Sería genial, gracias Elena.
Roberto Silva: Excelente. Carmen, ¿qué tienes para hoy?
Carmen Vega: Voy a continuar con el diseño de la nueva interfaz. Necesito feedback del cliente.
Roberto Silva: Te conseguiré una reunión con ellos esta tarde. ¿Te parece bien a las 4?
Carmen Vega: Perfecto. También quiero mencionar que necesitamos actualizar las librerías.
Roberto Silva: Buen punto. Diego, ¿puedes encargarte de eso cuando resuelvas el bug?
Diego Castro: Claro, lo agrego a mi lista.
Roberto Silva: Yo por mi parte, tengo reunión con el product owner para definir el siguiente sprint.
Elena Martín: Roberto, no olvides mencionar el tema de las estimaciones.
Roberto Silva: Anotado. ¿Algo más que deba llevar a esa reunión?
Carmen Vega: Sí, el tema del ambiente de staging. Necesitamos más recursos.
Roberto Silva: También lo anoto. Bueno equipo, ¿algún otro tema?
Diego Castro: Solo recordar que mañana es el code review de mi feature.
Roberto Silva: Cierto. Elena y Carmen, ¿pueden participar?
Elena Martín: Sí, cuenta conmigo.
Carmen Vega: Yo también.
Roberto Silva: Perfecto. Entonces nos vemos mañana a las 10 para el review. Buen día a todos.""",
        "speaker_transcript": """[Hablante 1] Buenos días equipo. Empecemos con el daily. Elena, ¿cómo vas?
[Hablante 2] Buenos días. Ayer terminé la integración con la API de pagos. Hoy voy a trabajar en las pruebas unitarias.
[Hablante 1] Genial. ¿Algún bloqueador?
[Hablante 2] No, todo bien por ahora.
[Hablante 1] Perfecto. Diego, tu turno.
[Hablante 3] Hola. Estoy atascado con el bug del módulo de reportes. Necesito ayuda.
[Hablante 1] ¿Qué tipo de ayuda necesitas?
[Hablante 3] No encuentro dónde está el error. Ya revisé el código tres veces.
[Hablante 2] Yo puedo ayudarte después del standup.
[Hablante 3] Sería genial, gracias Elena.
[Hablante 1] Excelente. Carmen, ¿qué tienes para hoy?
[Hablante 4] Voy a continuar con el diseño de la nueva interfaz. Necesito feedback del cliente.
[Hablante 1] Te conseguiré una reunión con ellos esta tarde. ¿Te parece bien a las 4?
[Hablante 4] Perfecto. También quiero mencionar que necesitamos actualizar las librerías.
[Hablante 1] Buen punto. Diego, ¿puedes encargarte de eso cuando resuelvas el bug?
[Hablante 3] Claro, lo agrego a mi lista.
[Hablante 1] Yo por mi parte, tengo reunión con el product owner para definir el siguiente sprint.
[Hablante 2] Roberto, no olvides mencionar el tema de las estimaciones.
[Hablante 1] Anotado. ¿Algo más que deba llevar a esa reunión?
[Hablante 4] Sí, el tema del ambiente de staging. Necesitamos más recursos.
[Hablante 1] También lo anoto. Bueno equipo, ¿algún otro tema?
[Hablante 3] Solo recordar que mañana es el code review de mi feature.
[Hablante 1] Cierto. Elena y Carmen, ¿pueden participar?
[Hablante 2] Sí, cuenta conmigo.
[Hablante 4] Yo también.
[Hablante 1] Perfecto. Entonces nos vemos mañana a las 10 para el review. Buen día a todos.""",
        "duration": 420,  # 7 minutes
        "speakers": ["Roberto Silva", "Elena Martín", "Diego Castro", "Carmen Vega"],
        "expected_tasks": [
            {
                "description": "Trabajar en pruebas unitarias de API de pagos",
                "assignee": "Elena",
                "assigned_by": "Elena"
            },
            {
                "description": "Ayudar a Diego con bug del módulo de reportes",
                "assignee": "Elena",
                "assigned_by": "Diego"
            },
            {
                "description": "Conseguir reunión con cliente para feedback",
                "assignee": "Roberto",
                "due_date": "hoy a las 4",
                "assigned_by": "Carmen"
            },
            {
                "description": "Actualizar las librerías",
                "assignee": "Diego",
                "assigned_by": "Roberto"
            }
        ]
    }
}


def get_conversation_by_type(conv_type: str) -> Dict:
    """Get a specific conversation type."""
    return CONVERSATIONS.get(conv_type, {})


def get_all_conversations() -> List[Dict]:
    """Get all conversations as a list."""
    return list(CONVERSATIONS.values())


def create_whisper_style_output(conversation: Dict) -> str:
    """Create a Whisper-style transcript output."""
    timestamp = datetime.now().strftime("%d %b %Y")
    duration_min = conversation['duration'] // 60
    duration_sec = conversation['duration'] % 60
    
    output = f"""{timestamp}
{conversation['title']} - Transcripción
00:00:00

{conversation['raw_transcript']}

La transcripción finalizó después de {duration_min:02d}:{duration_sec:02d}

Esta transcripción editable se ha generado por ordenador y puede contener errores.
"""
    return output


def get_sample_whisper_formats() -> Dict[str, str]:
    """Get various Whisper output format samples."""
    return {
        "with_timestamps": """[00:00:00] María: Buenos días equipo.
[00:00:03] Carlos: Buenos días María.
[00:00:05] María: Vamos a revisar los pendientes.""",
        
        "with_uncertainty": """María: Buenos días equipo.
Carlos: Buenos días [inaudible].
María: Vamos a revisar los... eh... pendientes del proyecto.""",
        
        "with_overlapping": """María: Entonces necesitamos--
Carlos: [interrumpiendo] Perdón María, ¿puedo agregar algo?
María: Sí, claro, adelante.""",
        
        "with_background_noise": """María: Buenos días equipo. [ruido de fondo]
Carlos: Buenos días María. [sonido de teclado]
María: Vamos a revisar los pendientes. [puerta cerrándose]"""
    }


# Test data for edge cases
EDGE_CASES = {
    "multiple_speakers_same_line": {
        "speaker_transcript": "[Hablante 1] Necesito que [Hablante 2] Sí, yo me encargo [Hablante 1] perfecto, gracias.",
        "duration": 30,
        "speakers": ["Hablante 1", "Hablante 2"]
    },
    
    "unclear_assignments": {
        "speaker_transcript": """[Hablante 1] Hay que hacer el reporte mensual.
[Hablante 2] Sí, es importante.
[Hablante 3] Totalmente de acuerdo.
[Hablante 1] Alguien debería encargarse de eso.""",
        "duration": 60,
        "speakers": ["Hablante 1", "Hablante 2", "Hablante 3"]
    },
    
    "informal_conversation": {
        "speaker_transcript": """[Hablante 1] ¿Viste el partido ayer?
[Hablante 2] Sí, estuvo increíble.
[Hablante 1] Por cierto, no olvides lo del cliente.
[Hablante 2] Ah sí, lo llamo en la tarde.""",
        "duration": 90,
        "speakers": ["Hablante 1", "Hablante 2"]
    }
}